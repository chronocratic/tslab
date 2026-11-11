"""``tslab study submit``: plans, writes, launches and records study runs."""

import dataclasses
from importlib import import_module
import json
import os
from pathlib import Path, PurePosixPath
import signal
import subprocess
import sys
import threading
import time
from typing import cast, TYPE_CHECKING

import pytest

from tests.lab.unit.study.helpers import (
    chained_context,
    chained_study,
    context,
    jobs_of,
    no_sleep,
    study_directory,
)
from tests.support.fake_backend import FakeBackend
from tslab.lab.core.status import utc_now_iso
from tslab.lab.enums import AfterCondition, JobPhase, StudyState
from tslab.lab.study.code_snapshot import CodeSnapshot
from tslab.lab.study.commands import submit, submit_runs
from tslab.lab.study.context import Context, load_context, Selection
from tslab.lab.study.launchers.slurm.records import SlurmJobRecord
from tslab.lab.study.records import JobKind, JobRecord, ProgressSnapshot, read_jobs, TaskRecord

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tslab.lab.study.plan import JobGroup

_SUBMIT_MODULE = import_module('tslab.lab.study.commands.submit')

# why: the small model has no ``depth`` field, so a study that sets
# ``model.depth`` for it does not load; the parse-bad study leaves the
# required ``data`` field unset, so its runs fail validation instead.
LOAD_BAD_STUDIES = """
import sys

import toy_script
from tslab.lab.study import Grid, Study

study = Study(
    name='s',
    grids={
        'g': Grid(
            short_name='g1',
            command=(sys.executable, 'toy_script.py'),
            config=toy_script.ToyConfig,
            axes={
                'model': {
                    'small': {'model': 'small', 'model.depth': 7},
                    'large': {'model': 'large'},
                },
                'seed': {'1': {'seed': 1}},
            },
        )
    },
)
"""

PARSE_BAD_STUDIES = """
import sys

import toy_script
from tslab.lab.study import Grid, Study

study = Study(
    name='s',
    grids={
        'g': Grid(
            short_name='g1',
            command=(sys.executable, 'toy_script.py'),
            config=toy_script.ToyConfig,
            axes={
                'model': {'small': {'model': 'small'}, 'large': {'model': 'large'}},
                'seed': {'1': {'seed': 1}, '2': {'seed': 2}},
            },
            # why: clears the data setting, so every run omits a required field
            run_settings={},
        )
    },
)
"""

# why: the plain toy run ends before a signal could reach it; this one stays
# open in the run context, so a stop signal records the run as crashed.
KEEP_OPEN_SCRIPT = '''
"""A toy run that stays open until a signal arrives."""

import time

import tyro

import tslab.lab

import toy_script


def main() -> None:
    config = tyro.cli(toy_script.ToyConfig, config=(tyro.conf.CascadeSubcommandArgs,))
    with tslab.lab.Run(config=config, root=config.output_root, relative_path=config.run_path):
        time.sleep(10)


if __name__ == '__main__':
    main()
'''

KEEP_OPEN_STUDIES = '''
"""The toy study with runs that stay open until a signal arrives."""

import sys

import toy_script
from tslab.lab.study import DEFAULT_RUN_SETTINGS, Grid, Study

study = Study(
    name='s',
    grids={
        'g': Grid(
            short_name='g1',
            command=(sys.executable, 'toy_keep_open.py'),
            config=toy_script.ToyConfig,
            axes={
                'model': {
                    'small': {'model': 'small'},
                    'large': {'model': 'large'},
                },
                'seed': {'1': {'seed': 1}, '2': {'seed': 2}},
            },
            run_settings={**DEFAULT_RUN_SETTINGS, 'data': 'data'},
        )
    },
)
'''


def run_toy(*, project: Path, model: str, seed: int) -> None:
    """Run one toy run for real, so its status.json is a real one."""
    subprocess.run(  # noqa: S603 # why: the toy script of the project, not external input
        [
            sys.executable,
            'toy_script.py',
            f'model:{model}',
            '--seed',
            str(seed),
            # why: the runs of the study live under <output_root>/<study>
            '--output-root',
            str(project / 'runs' / 's'),
            '--run-path',
            f'g/{model}/{seed}',
            '--data',
            str(project / 'data'),
        ],
        cwd=project,
        check=True,
        capture_output=True,
    )


def test_submit_launches_and_records(project: Path, fake_backend: FakeBackend) -> None:
    code = submit(
        context=context(project=project, machine_file='cluster.toml'),
        include_done=False,
        dry_run=False,
        extra_args=(),
        sleep=no_sleep,
    )
    assert code == 0
    assert len(fake_backend.calls('launch')) == 1
    jobs = jobs_of(project=project)
    assert len(jobs) == 1
    assert [(t.run_path.as_posix(), t.index) for t in jobs[0].tasks] == [
        ('g/small/1', 0),
        ('g/small/2', 1),
        ('g/large/1', 2),
        ('g/large/2', 3),
    ]
    assert all(t.attempt == 1 for t in jobs[0].tasks)


def test_submit_orders_upstream_groups_first(project: Path, fake_backend: FakeBackend) -> None:
    ctx = chained_context(project=project, study=chained_study())

    code = submit(context=ctx, include_done=False, dry_run=False, extra_args=(), sleep=no_sleep)

    assert code == 0
    groups = cast('Sequence[JobGroup]', fake_backend.calls('launch')[0]['groups'])
    assert [group.grid for group in groups] == ['pretrain', 'probe', 'probe']


def test_submit_uses_job_records_read_under_lock_for_targets(
    project: Path, fake_backend: FakeBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = chained_context(
        project=project, study=chained_study(), grids=('probe',), machine_file='cluster.toml'
    )
    upstream_path = PurePosixPath('pretrain/a')
    upstream = dataclasses.replace(ctx.upstream_reports[upstream_path], state=StudyState.RUNNING)
    ctx = dataclasses.replace(ctx, upstream_reports={upstream_path: upstream})
    task = TaskRecord(
        index=3,
        grid='pretrain',
        run_path=upstream_path,
        command=(),
        extra_args=(),
        resources={},
        attempt=1,
        progress=ProgressSnapshot(done_instances=0, newest_checkpoint_mtime=None),
    )
    job = SlurmJobRecord(
        kind=JobKind.RUN,
        submission_id='prior',
        time=utc_now_iso(),
        launcher=ctx.machine.launcher,
        job_id='41',
        job_file=1,
        tasks=(task,),
    )
    monkeypatch.setattr(
        _SUBMIT_MODULE,
        'read_jobs',
        lambda *, path: [job] if path.name == 'submissions.jsonl' else [],
    )

    code = submit(context=ctx, include_done=False, dry_run=False, extra_args=(), sleep=no_sleep)

    assert code == 0
    launches = fake_backend.calls('launch')
    assert len(launches) == 1
    groups = cast('Sequence[JobGroup]', launches[0]['groups'])
    assert groups[0].after[0].job_tasks == ('41_3',)


def test_ignore_after_submits_runs_blocked_by_upstream(
    project: Path, fake_backend: FakeBackend
) -> None:
    ctx = chained_context(project=project, study=chained_study(), grids=('probe',))
    upstream_path = PurePosixPath('pretrain/a')
    upstream = dataclasses.replace(ctx.upstream_reports[upstream_path], state=StudyState.FAILED)
    ctx = dataclasses.replace(ctx, upstream_reports={upstream_path: upstream})

    code = submit(context=ctx, include_done=False, dry_run=False, extra_args=(), sleep=no_sleep)

    assert code == 0
    assert fake_backend.calls('launch') == []
    code = submit(
        context=ctx,
        include_done=False,
        dry_run=False,
        extra_args=(),
        sleep=no_sleep,
        ignore_after=True,
    )

    assert code == 0
    (group,) = cast('Sequence[JobGroup]', fake_backend.calls('launch')[0]['groups'])
    assert group.after == ()


def test_started_dependency_is_rejected_on_local_machine(
    project: Path, fake_backend: FakeBackend, caplog: pytest.LogCaptureFixture
) -> None:
    ctx = chained_context(
        project=project, study=chained_study(condition=AfterCondition.STARTED), grids=('probe',)
    )

    with caplog.at_level('ERROR'):
        code = submit(context=ctx, include_done=False, dry_run=False, extra_args=(), sleep=no_sleep)

    assert code == 1
    assert "grid 'probe': condition 'started' needs SLURM" in caplog.text
    assert fake_backend.calls('launch') == []


# why: the pretrain of the local chain study fails on purpose through the
# environment, so one launch covers the done, failed and failed-instance cases.
CHAIN_PRE_SCRIPT = '''
"""The pretrain of the toy chained study."""

from dataclasses import dataclass
import os
from pathlib import Path

import tyro

import tslab.lab
from tslab.lab.core.status import InstanceRecord, RunStatusSnapshot, utc_now_iso, write_status
from tslab.lab.enums import InstanceState, RunState


@dataclass(frozen=True, kw_only=True)
class ChainPreConfig:
    """The config of one pretrain run."""

    dataset: str
    data: Path
    output_root: Path = Path('.')
    run_path: Path = Path('run')


def main() -> None:
    config = tyro.cli(ChainPreConfig)
    if os.environ.get('TOY_CHAIN_FAIL'):
        with tslab.lab.Run(config=config, root=config.output_root, relative_path=config.run_path):
            raise RuntimeError('pretrain fails on purpose')
    folder = config.output_root / config.run_path
    if os.environ.get('TOY_CHAIN_TIME_LIMIT'):
        # the first attempt saves a checkpoint and stops at the time limit, the next finishes
        limited = not (folder / 'train' / 'last.ckpt').exists()
        (folder / 'train').mkdir(parents=True, exist_ok=True)
        (folder / 'train' / 'last.ckpt').write_text('checkpoint')
        record = (
            InstanceRecord(state=InstanceState.FAILED, error='JobTimeLimitSignal: time limit')
            if limited
            else InstanceRecord(state=InstanceState.DONE)
        )
        write_status(
            path=folder / 'status.json',
            snapshot=RunStatusSnapshot(
                run_state=RunState.RUNNING if limited else RunState.FINISHED,
                updated_at=utc_now_iso(),
                instances={'train': record},
            ),
        )
        raise SystemExit(1 if limited else 0)
    if os.environ.get('TOY_CHAIN_FAIL_INSTANCE'):
        folder = config.output_root / config.run_path
        folder.mkdir(parents=True, exist_ok=True)
        write_status(
            path=folder / 'status.json',
            snapshot=RunStatusSnapshot(
                run_state=RunState.FINISHED,
                updated_at=utc_now_iso(),
                instances={
                    'train': InstanceRecord(
                        state=InstanceState.FAILED, error='an instance fails on purpose'
                    )
                },
            ),
        )
        return
    with tslab.lab.Run(config=config, root=config.output_root, relative_path=config.run_path):
        pass


if __name__ == '__main__':
    main()
'''

CHAIN_PROBE_SCRIPT = '''
"""The probe of the toy chained study; needs the upstream run folder."""

from dataclasses import dataclass
from pathlib import Path

import tyro

import tslab.lab


@dataclass(frozen=True, kw_only=True)
class ChainProbeConfig:
    """The config of one probe run."""

    dataset: str
    data: Path
    checkpoint: Path
    output_root: Path = Path('.')
    run_path: Path = Path('run')


def main() -> None:
    config = tyro.cli(ChainProbeConfig)
    if not (config.checkpoint / 'status.json').is_file():
        raise SystemExit(f'upstream run folder missing: {config.checkpoint}')
    with tslab.lab.Run(config=config, root=config.output_root, relative_path=config.run_path):
        pass


if __name__ == '__main__':
    main()
'''

CHAIN_OK_SCRIPT = '''
"""An always-succeeding toy run, for the grid without a dependency."""

from dataclasses import dataclass
from pathlib import Path

import tyro

import tslab.lab


@dataclass(frozen=True, kw_only=True)
class ChainOkConfig:
    """The config of one independent run."""

    dataset: str
    data: Path
    output_root: Path = Path('.')
    run_path: Path = Path('run')


def main() -> None:
    config = tyro.cli(ChainOkConfig)
    with tslab.lab.Run(config=config, root=config.output_root, relative_path=config.run_path):
        pass


if __name__ == '__main__':
    main()
'''

CHAIN_STUDIES = '''
"""The toy chained study: pretrain, probe and an independent grid."""

import sys

import toy_chain_ok
import toy_chain_pre
import toy_chain_probe
from tslab.lab.study import DEFAULT_RUN_SETTINGS, After, Grid, Study

study = Study(
    name='chain',
    grids={
        'pretrain': Grid(
            short_name='pre',
            command=(sys.executable, 'toy_chain_pre.py'),
            config=toy_chain_pre.ChainPreConfig,
            axes={'dataset': {'a': {'dataset': 'a'}, 'b': {'dataset': 'b'}}},
            run_settings={**DEFAULT_RUN_SETTINGS, 'data': 'data'},
        ),
        'probe': Grid(
            short_name='probe',
            command=(sys.executable, 'toy_chain_probe.py'),
            config=toy_chain_probe.ChainProbeConfig,
            axes={'dataset': {'a': {'dataset': 'a'}, 'b': {'dataset': 'b'}}},
            after=(After(grid='pretrain'),),
            run_settings={
                **DEFAULT_RUN_SETTINGS,
                'data': 'data',
                'checkpoint': '{after:pretrain}',
            },
        ),
        'other': Grid(
            short_name='oth',
            command=(sys.executable, 'toy_chain_ok.py'),
            config=toy_chain_ok.ChainOkConfig,
            axes={'dataset': {'c': {'dataset': 'c'}}},
            run_settings={**DEFAULT_RUN_SETTINGS, 'data': 'data'},
        ),
    },
)
'''


def write_chain_project(*, project: Path) -> None:
    """The toy project plus the chained toy study and its scripts."""
    (project / 'toy_chain_pre.py').write_text(CHAIN_PRE_SCRIPT.lstrip('\n'))
    (project / 'toy_chain_probe.py').write_text(CHAIN_PROBE_SCRIPT.lstrip('\n'))
    (project / 'toy_chain_ok.py').write_text(CHAIN_OK_SCRIPT.lstrip('\n'))
    (project / 'toy_chain_studies.py').write_text(CHAIN_STUDIES.lstrip('\n'))


def chain_context(*, project: Path) -> Context:
    """The context of the chained toy study on the local machine of the project."""
    return load_context(
        reference='toy_chain_studies:study',
        machine_path=project / 'local.toml',
        selection=Selection(),
        project_root=Path.cwd(),
        local_cpus=1,
        environment=os.environ,
    )


def chain_jobs(*, project: Path) -> list[JobRecord]:
    """The job records of the chained toy study."""
    return read_jobs(path=project / 'runs' / 'chain' / '.study' / 'submissions.jsonl')


def chain_states(*, project: Path) -> dict[str, StudyState]:
    """The state of every run of the chained toy study, by run path."""
    reports = chain_context(project=project).reports
    return {report.run.run_path.as_posix(): report.state for report in reports}


@pytest.mark.slow
def test_local_chain_runs_probe_after_pretrain(project: Path) -> None:
    write_chain_project(project=project)

    code = submit(
        context=chain_context(project=project),
        include_done=False,
        dry_run=False,
        extra_args=(),
        sleep=no_sleep,
    )

    assert code == 0
    tasks = [task for job in chain_jobs(project=project) for task in job.tasks]
    assert {task.run_path.as_posix() for task in tasks} == {
        'pretrain/a',
        'pretrain/b',
        'probe/a',
        'probe/b',
        'other/c',
    }
    probe_a = next(task for task in tasks if task.run_path.as_posix() == 'probe/a')
    # why: the placeholder of the upstream run is in the command; the probe
    # refuses to run without the upstream status file
    assert 'chain/pretrain/a' in ' '.join(probe_a.command)
    assert chain_states(project=project) == {
        'pretrain/a': StudyState.DONE,
        'pretrain/b': StudyState.DONE,
        'probe/a': StudyState.DONE,
        'probe/b': StudyState.DONE,
        'other/c': StudyState.DONE,
    }


@pytest.mark.slow
def test_local_failed_upstream_skips_probe(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_chain_project(project=project)
    monkeypatch.setenv('TOY_CHAIN_FAIL', '1')

    code = submit(
        context=chain_context(project=project),
        include_done=False,
        dry_run=False,
        extra_args=(),
        sleep=no_sleep,
    )

    assert code == 0
    tasks = [task for job in chain_jobs(project=project) for task in job.tasks]
    assert {task.run_path.as_posix() for task in tasks} == {'pretrain/a', 'pretrain/b', 'other/c'}
    assert not (project / 'runs' / 'chain' / 'probe').exists()
    assert chain_states(project=project) == {
        'pretrain/a': StudyState.FAILED,
        'pretrain/b': StudyState.FAILED,
        'probe/a': StudyState.DEPENDENCY_FAILED,
        'probe/b': StudyState.DEPENDENCY_FAILED,
        'other/c': StudyState.DONE,
    }


@pytest.mark.slow
def test_local_failed_instance_blocks_ok_probe(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_chain_project(project=project)
    monkeypatch.setenv('TOY_CHAIN_FAIL_INSTANCE', '1')

    code = submit(
        context=chain_context(project=project),
        include_done=False,
        dry_run=False,
        extra_args=(),
        sleep=no_sleep,
    )

    assert code == 0
    tasks = [task for job in chain_jobs(project=project) for task in job.tasks]
    assert {task.run_path.as_posix() for task in tasks} == {'pretrain/a', 'pretrain/b', 'other/c'}
    assert not (project / 'runs' / 'chain' / 'probe').exists()
    assert chain_states(project=project) == {
        'pretrain/a': StudyState.FAILED,
        'pretrain/b': StudyState.FAILED,
        'probe/a': StudyState.DEPENDENCY_FAILED,
        'probe/b': StudyState.DEPENDENCY_FAILED,
        'other/c': StudyState.DONE,
    }


@pytest.mark.slow
def test_local_chained_resubmit_reruns_probe_after_upstream_time_limit(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_chain_project(project=project)
    machine = project / 'local.toml'
    machine.write_text(machine.read_text() + '\n[followup]\nchained_retry = "resubmit"\n')
    monkeypatch.setenv('TOY_CHAIN_TIME_LIMIT', '1')

    code = submit(
        context=chain_context(project=project),
        include_done=False,
        dry_run=False,
        extra_args=(),
        sleep=no_sleep,
    )

    assert code == 0
    tasks = [task for job in chain_jobs(project=project) for task in job.tasks]
    attempts = {(task.run_path.as_posix(), task.attempt) for task in tasks}
    assert {('pretrain/a', 1), ('pretrain/a', 2), ('probe/a', 1), ('probe/b', 1)} <= attempts
    assert chain_states(project=project) == {
        'pretrain/a': StudyState.DONE,
        'pretrain/b': StudyState.DONE,
        'probe/a': StudyState.DONE,
        'probe/b': StudyState.DONE,
        'other/c': StudyState.DONE,
    }


def test_submit_runs_records_the_code_snapshot(project: Path, fake_backend: FakeBackend) -> None:
    snapshot = CodeSnapshot(
        folder=project / 'runs' / 's' / '.tslab-code-snapshots' / '0123456789abcdef',
        hash='0123456789abcdef',
        commit='a' * 40,
        dirty=False,
        untracked_copied=False,
    )
    ctx = context(project=project, machine_file='cluster.toml')
    code, submitted = submit_runs(
        context=ctx,
        reports=ctx.reports,
        dry_run=False,
        extra_args={report.run.run_path: () for report in ctx.reports},
        sleep=no_sleep,
        code_snapshots={report.run.run_path: snapshot for report in ctx.reports},
    )
    assert code == 0
    assert submitted
    assert all(record.code_snapshot == snapshot for record in submitted)
    assert jobs_of(project=project)[-1].code_snapshot == snapshot


def test_submit_skips_done_and_live_runs(project: Path, fake_backend: FakeBackend) -> None:
    first = submit(
        context=context(project=project, machine_file='cluster.toml'),
        include_done=False,
        dry_run=False,
        extra_args=(),
        sleep=no_sleep,
    )
    assert first == 0
    fake_backend.set_states(
        phases={
            'job-0_0': JobPhase.QUEUED,
            'job-0_1': JobPhase.RUNNING,
            'job-0_2': JobPhase.COMPLETED,
            'job-0_3': JobPhase.INTERRUPTED,
        }
    )
    run_toy(project=project, model='large', seed=1)
    code = submit(
        context=context(project=project, machine_file='cluster.toml'),
        include_done=False,
        dry_run=False,
        extra_args=(),
        sleep=no_sleep,
    )
    assert code == 0
    jobs = jobs_of(project=project)
    assert [t.run_path.as_posix() for t in jobs[1].tasks] == ['g/large/2']
    task = jobs[1].tasks[0]
    assert task.attempt == 2
    # why: the run keeps the array index of its first submission
    assert task.index == 3


def test_include_done_resubmits_done_runs(project: Path, fake_backend: FakeBackend) -> None:
    assert (
        submit(
            context=context(project=project, machine_file='cluster.toml'),
            include_done=False,
            dry_run=False,
            extra_args=(),
            sleep=no_sleep,
        )
        == 0
    )
    fake_backend.set_states(
        phases={
            'job-0_0': JobPhase.COMPLETED,
            'job-0_1': JobPhase.COMPLETED,
            'job-0_2': JobPhase.COMPLETED,
            'job-0_3': JobPhase.COMPLETED,
        }
    )
    run_toy(project=project, model='large', seed=1)
    code = submit(
        context=context(project=project, machine_file='cluster.toml'),
        include_done=False,
        dry_run=False,
        extra_args=(),
        sleep=no_sleep,
    )
    assert code == 0
    jobs = jobs_of(project=project)
    # why: the done run is skipped, the runs without a folder are failed, not done
    assert [t.run_path.as_posix() for t in jobs[1].tasks] == ['g/small/1', 'g/small/2', 'g/large/2']
    code = submit(
        context=context(project=project, machine_file='cluster.toml'),
        include_done=True,
        dry_run=False,
        extra_args=(),
        sleep=no_sleep,
    )
    assert code == 0
    jobs = jobs_of(project=project)
    assert [t.run_path.as_posix() for t in jobs[2].tasks] == [
        'g/small/1',
        'g/small/2',
        'g/large/1',
        'g/large/2',
    ]
    assert len(fake_backend.calls('launch')) == 3


def test_submit_stops_before_writing_when_a_run_does_not_parse(
    project: Path, fake_backend: FakeBackend, caplog: pytest.LogCaptureFixture
) -> None:
    (project / 'bad_load.py').write_text(LOAD_BAD_STUDIES.lstrip('\n'))
    with pytest.raises(ValueError, match='not fields'):
        load_context(
            reference='bad_load:study',
            machine_path=project / 'cluster.toml',
            selection=Selection(),
            project_root=Path.cwd(),
            local_cpus=1,
            environment=os.environ,
        )
    (project / 'bad_parse.py').write_text(PARSE_BAD_STUDIES.lstrip('\n'))
    with caplog.at_level('INFO'):
        code = submit(
            context=load_context(
                reference='bad_parse:study',
                machine_path=project / 'cluster.toml',
                selection=Selection(),
                project_root=Path.cwd(),
                local_cpus=1,
                environment=os.environ,
            ),
            include_done=False,
            dry_run=False,
            extra_args=(),
            sleep=no_sleep,
        )
    assert code == 1
    assert fake_backend.calls('launch') == []
    assert not (project / 'runs' / 's' / '.study' / 'jobs').exists()
    assert 'g/small/1' in caplog.text


def test_dry_run_prints_plan_and_records_nothing(
    project: Path, fake_backend: FakeBackend, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level('INFO'):
        code = submit(
            context=context(project=project, machine_file='cluster.toml'),
            include_done=False,
            dry_run=True,
            extra_args=(),
            sleep=no_sleep,
        )
    assert code == 0
    assert [call['dry_run'] for call in fake_backend.calls('launch')] == [True]
    assert not (project / 'runs' / 's' / '.study' / 'submissions.jsonl').exists()
    text = ' '.join(caplog.text.split())
    assert 'g1 4 fake resources' in text
    assert 'toy_script.py model:small --seed 1' in text


def write_local_machine(project: Path, *, local: str = '') -> None:
    """The project's local machine, with a ``[local]`` table when given one."""
    text = f'launcher = "local"\noutput_root = "{(project / "runs").as_posix()}"\n'
    if local:
        text += f'\n[local]\n{local}\n'
    (project / 'local.toml').write_text(text)


def fake_nvidia_smi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, busy: tuple[str, ...] = ()
) -> None:
    """A fake nvidia-smi on PATH; GPUs 0 and 1, the given ones busy."""
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / 'nvidia-smi').write_text(
        '#!/usr/bin/env python3\n'
        'import os\n'
        'import sys\n'
        'args = " ".join(sys.argv[1:])\n'
        'busy = set(os.environ.get("FAKE_BUSY", "").split(","))\n'
        'if "--query-compute-apps=gpu_uuid" in args:\n'
        '    for index in "0 1".split():\n'
        '        if index in busy:\n'
        '            print("GPU-" + index)\n'
        'elif "--query-gpu=index,uuid" in args:\n'
        '    for index in "0 1".split():\n'
        '        print(index + ",GPU-" + index)\n'
    )
    (bin_dir / 'nvidia-smi').chmod(0o755)
    monkeypatch.setenv('PATH', f'{bin_dir}:{os.environ["PATH"]}')
    monkeypatch.setenv('FAKE_BUSY', ','.join(busy))


def fake_caffeinate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake caffeinate on PATH that writes its arguments; the argument file."""
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir(exist_ok=True)
    log = tmp_path / 'caffeinate.args'
    (bin_dir / 'caffeinate').write_text(
        '#!/usr/bin/env python3\n'
        'import os\n'
        'import sys\n'
        'import time\n'
        'with open(os.environ["FAKE_CAFFEINATE_LOG"], "w") as f:\n'
        '    f.write(" ".join(sys.argv[1:]))\n'
        'time.sleep(30)\n'
    )
    (bin_dir / 'caffeinate').chmod(0o755)
    monkeypatch.setenv('PATH', f'{bin_dir}:{os.environ["PATH"]}')
    monkeypatch.setenv('FAKE_CAFFEINATE_LOG', str(log))
    return log


def gpu_ids_of_usage_files(*, project: Path) -> list[str | None]:
    """The gpu_id of every usage file of the submitted tasks, in task order."""
    jobs = jobs_of(project=project)
    assert len(jobs) == 1
    logs = study_directory(project=project) / '.study' / 'logs'
    return [
        json.loads((logs / f's_g1_{task.index}.out.usage.json').read_text())['gpu_id']
        for task in jobs[0].tasks
    ]


def test_auto_gpus_from_cuda_visible_devices(
    project: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_local_machine(project, local='gpus = "auto"\n')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1,3')
    with caplog.at_level('INFO'):
        code = submit(
            context=context(project=project, machine_file='local.toml'),
            include_done=False,
            dry_run=False,
            extra_args=(),
            sleep=no_sleep,
        )
    assert code == 0
    assert 'local GPUs: 1,3' in caplog.text
    assert set(gpu_ids_of_usage_files(project=project)) == {'1', '3'}


def test_auto_gpus_uuid_fails_submit(
    project: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_local_machine(project, local='gpus = "auto"\n')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-abc')
    with caplog.at_level('INFO'):
        code = submit(
            context=context(project=project, machine_file='local.toml'),
            include_done=False,
            dry_run=False,
            extra_args=(),
            sleep=no_sleep,
        )
    assert code == 1
    assert 'local GPUs' not in caplog.text
    assert not (project / 'runs' / 's' / '.study' / 'submissions.jsonl').exists()


def test_skip_busy_gpus_leaves_busy_out(
    project: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_nvidia_smi(tmp_path, monkeypatch, busy=('1',))
    write_local_machine(project, local='gpus = [0, 1]\nskip_busy_gpus = true\n')
    with caplog.at_level('INFO'):
        code = submit(
            context=context(project=project, machine_file='local.toml'),
            include_done=False,
            dry_run=False,
            extra_args=(),
            sleep=no_sleep,
        )
    assert code == 0
    assert 'local GPUs: 0' in caplog.text
    assert 'local GPUs: 0,1' not in caplog.text
    assert gpu_ids_of_usage_files(project=project) == ['0'] * 4


def test_skip_busy_gpus_all_busy_fails(
    project: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_nvidia_smi(tmp_path, monkeypatch, busy=('1',))
    write_local_machine(project, local='gpus = [1]\nskip_busy_gpus = true\n')
    with caplog.at_level('INFO'):
        code = submit(
            context=context(project=project, machine_file='local.toml'),
            include_done=False,
            dry_run=False,
            extra_args=(),
            sleep=no_sleep,
        )
    assert code == 1
    assert 'all local GPUs are busy: 1' in caplog.text


def test_auto_gpus_with_max_parallel_warns(
    project: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_local_machine(project, local='gpus = "auto"\nmax_parallel = 2\n')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')
    with caplog.at_level('WARNING'):
        code = submit(
            context=context(project=project, machine_file='local.toml'),
            include_done=False,
            dry_run=False,
            extra_args=(),
            sleep=no_sleep,
        )
    assert code == 0
    assert 'max_parallel is ignored' in caplog.text


def test_explicit_mps_runs_one_task_with_env_untouched(
    project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    xdg = tmp_path / 'xdg'
    monkeypatch.setenv('XDG_RUNTIME_DIR', str(xdg))
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    write_local_machine(project, local='gpus = ["mps"]\n')
    code = submit(
        context=context(project=project, machine_file='local.toml'),
        include_done=False,
        dry_run=False,
        extra_args=(),
        sleep=no_sleep,
    )
    assert code == 0
    assert gpu_ids_of_usage_files(project=project) == ['mps'] * 4
    assert (xdg / 'tslab-gpus' / 'gpu-mps.lock').is_file()


def test_keep_awake_helper_gets_launch_pid(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pids: list[int] = []

    def recording_start(*, pid: int, reason: str) -> None:
        del reason
        pids.append(pid)

    monkeypatch.setattr('tslab.lab.study.launchers.local.backend.start_keep_awake', recording_start)
    code = submit(
        context=context(project=project, machine_file='local.toml'),
        include_done=False,
        dry_run=False,
        extra_args=(),
        sleep=no_sleep,
    )
    assert code == 0
    assert pids == [os.getpid()]


def test_keep_awake_missing_tool_still_runs(
    project: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    monkeypatch.setenv('PATH', str(bin_dir))
    with caplog.at_level('WARNING'):
        code = submit(
            context=context(project=project, machine_file='local.toml'),
            include_done=False,
            dry_run=False,
            extra_args=(),
            sleep=no_sleep,
        )
    assert code == 0
    assert 'no caffeinate found' in caplog.text


def test_keep_awake_false_starts_no_helper(
    project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = fake_caffeinate(tmp_path, monkeypatch)
    write_local_machine(project, local='keep_awake = false\n')
    code = submit(
        context=context(project=project, machine_file='local.toml'),
        include_done=False,
        dry_run=False,
        extra_args=(),
        sleep=no_sleep,
    )
    assert code == 0
    assert not args.exists()


def _kill_when_a_run_starts(*, project: Path) -> None:
    """Send this process a SIGTERM as soon as the first toy run writes its status."""
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if next(iter(project.glob('runs/s/g/*/*/status.json')), None) is not None:
            os.kill(os.getpid(), signal.SIGTERM)
            return
        time.sleep(0.01)


def test_submit_returns_128_plus_signal(project: Path) -> None:
    (project / 'toy_keep_open.py').write_text(KEEP_OPEN_SCRIPT.lstrip('\n'))
    (project / 'toy_studies.py').write_text(KEEP_OPEN_STUDIES.lstrip('\n'))
    watcher = threading.Thread(
        target=_kill_when_a_run_starts, kwargs={'project': project}, daemon=True
    )
    watcher.start()
    code = submit(
        context=context(project=project, machine_file='local.toml'),
        include_done=False,
        dry_run=False,
        extra_args=(),
        sleep=no_sleep,
    )
    watcher.join(timeout=10)
    assert code == 128 + signal.SIGTERM
    reports = context(project=project, machine_file='local.toml').reports
    assert StudyState.INTERRUPTED in {report.state for report in reports}


@pytest.mark.slow
def test_local_submit_runs_everything(project: Path, caplog: pytest.LogCaptureFixture) -> None:
    first = submit(
        context=context(project=project, machine_file='local.toml'),
        include_done=False,
        dry_run=False,
        extra_args=(),
        sleep=no_sleep,
    )
    assert first == 0
    reports = load_context(
        reference='toy_studies:study',
        machine_path=project / 'local.toml',
        selection=Selection(),
        project_root=Path.cwd(),
        local_cpus=1,
        environment=os.environ,
    ).reports
    assert {report.state for report in reports} == {StudyState.DONE}
    with caplog.at_level('INFO'):
        code = submit(
            context=context(project=project, machine_file='local.toml'),
            include_done=False,
            dry_run=False,
            extra_args=(),
            sleep=no_sleep,
        )
    assert code == 0
    assert 'nothing to submit' in caplog.text
    assert len(jobs_of(project=project)) == 1

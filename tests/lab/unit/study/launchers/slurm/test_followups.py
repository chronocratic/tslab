"""Follow-up jobs: the automatic retry and the W&B sync behind the jobs of the study."""

import dataclasses
import importlib.util
import os
from pathlib import Path, PurePosixPath
import sys

import pytest

from tests.lab.unit.study.helpers import context, jobs_of, no_sleep, submit_study
from tests.support.fake_slurm import FakeSlurm
from tslab.lab.core.status import InstanceRecord, RunStatusSnapshot, utc_now_iso, write_status
from tslab.lab.enums import InstanceState, LauncherKind, RunState
from tslab.lab.study.commands import resubmit, submit
from tslab.lab.study.launchers.slurm.followups import queue_followups
from tslab.lab.study.launchers.slurm.records import SlurmJobRecord
from tslab.lab.study.machine import resolve_machine, write_register
from tslab.lab.study.records import append_job, JobKind, ProgressSnapshot, TaskRecord

# why: the shared toy project holds four runs; the auto-retry test needs six
STUDIES = '''
"""The studies of the toy project."""

import sys

import toy_script
from tslab.lab.study import DEFAULT_RUN_SETTINGS, Grid, Study

study = Study(
    name='s',
    grids={
        'g': Grid(
            short_name='g1',
            command=(sys.executable, 'toy_script.py'),
            config=toy_script.ToyConfig,
            axes={
                'model': {
                    'small': {'model': 'small'},
                    'large': {'model': 'large'},
                },
                'seed': {'1': {'seed': 1}, '2': {'seed': 2}, '3': {'seed': 3}},
            },
            run_settings={**DEFAULT_RUN_SETTINGS, 'data': 'data'},
        ),
    },
)
'''


def enable_retry(*, project: Path) -> None:
    """Let the automatic retry resubmit the runs that are not chained."""
    profile = project / 'cluster.toml'
    profile.write_text(profile.read_text() + '\n[followup]\nretry = "resubmit"\n')


def with_followups(*, project: Path) -> None:
    """The [followup] table, on top of the written cluster profile."""
    profile = project / 'cluster.toml'
    text = profile.read_text() + '\n[followup]\nretry = "resubmit"\nwandb_sync = true\n'
    profile.write_text(text)


def job_file_of(*, project: Path, job_file: int, suffix: str) -> Path:
    return project / 'runs' / 's' / '.study' / 'jobs' / f'{job_file}.{suffix}'


def bump_attempts(
    *, project: Path, run_path: str, index: int, up_to: int, uncounted: tuple[int, int] = (0, 0)
) -> None:
    """More attempts of one run, recorded like real submissions."""
    for attempt in range(2, up_to + 1):
        record = SlurmJobRecord(
            kind=JobKind.RUN,
            submission_id=f't{attempt}',
            time=utc_now_iso(),
            launcher=LauncherKind.SLURM,
            job_id=f'200{attempt}',
            job_file=attempt,
            tasks=(
                TaskRecord(
                    index=index,
                    grid='g',
                    run_path=PurePosixPath(run_path),
                    command=(sys.executable, 'toy_script.py'),
                    extra_args=(),
                    resources={},
                    attempt=attempt,
                    progress=ProgressSnapshot(done_instances=0, newest_checkpoint_mtime=None),
                    uncounted_attempts=uncounted[0],
                    uncounted_in_row=uncounted[1],
                ),
            ),
        )
        append_job(path=project / 'runs' / 's' / '.study' / 'submissions.jsonl', record=record)


@pytest.fixture(autouse=True)
def wandb_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The submitting environment has tslab.ml and wandb, as tslab[wandb] does."""
    real = importlib.util.find_spec
    present = importlib.util.spec_from_loader('present', loader=None)
    monkeypatch.setattr(
        importlib.util,
        'find_spec',
        lambda name, *args: present if name in {'tslab.ml', 'wandb'} else real(name, *args),
    )


@pytest.fixture
def project(project: Path) -> Path:
    folder = project
    (folder / 'toy_studies.py').write_text(STUDIES)
    return folder


def test_submit_queues_followups_after_the_arrays(project: Path, fake: FakeSlurm) -> None:
    with_followups(project=project)
    submit_study(project=project)
    assert len(fake.calls('sbatch')) == 3
    jobs = jobs_of(project=project)
    assert [job.kind for job in jobs] == ['run', 'followup', 'followup']
    retry, wbsync = jobs[1], jobs[2]
    assert not retry.tasks
    retry_script = job_file_of(
        project=project, job_file=retry.job_file, suffix='sbatch'
    ).read_text()
    assert '--job-name=s_retry' in retry_script
    assert '--dependency=afterany:1000' in retry_script
    assert '--array' not in retry_script
    assert '#SBATCH --time=00:30:00' in retry_script
    assert '#SBATCH --signal=USR2@120' not in retry_script
    assert '--gres' not in retry_script
    retry_tasks = job_file_of(project=project, job_file=retry.job_file, suffix='tasks').read_text()
    machine = (project / 'cluster.toml').as_posix()
    assert retry_tasks == (
        f'0 1 uv run tslab study resubmit toy_studies:study --machine {machine} --auto\n'
    )
    wbsync_script = job_file_of(
        project=project, job_file=wbsync.job_file, suffix='sbatch'
    ).read_text()
    assert '--job-name=s_wbsync' in wbsync_script
    wbsync_tasks = job_file_of(
        project=project, job_file=wbsync.job_file, suffix='tasks'
    ).read_text()
    assert wbsync_tasks == f'0 1 uv run tslab wandb-sync {(project / "runs" / "s").as_posix()}\n'


def test_chained_retry_alone_queues_the_retry_followup(project: Path, fake: FakeSlurm) -> None:
    profile = project / 'cluster.toml'
    profile.write_text(profile.read_text() + '\n[followup]\nchained_retry = "resubmit"\n')
    code, records = queue_followups(
        context=context(project=project, machine_file='cluster.toml'),
        job_ids=['101'],
        now=utc_now_iso,
        sleep=no_sleep,
    )

    assert code == 0
    assert len(records) == 1
    record = records[0]
    assert isinstance(record, SlurmJobRecord)
    assert record.job_name == 's_retry'


def test_submit_records_the_job_name_of_every_slurm_job(project: Path, fake: FakeSlurm) -> None:
    with_followups(project=project)
    submit_study(project=project)
    jobs = jobs_of(project=project)
    assert all(isinstance(job, SlurmJobRecord) for job in jobs)
    names = [job.job_name for job in jobs if isinstance(job, SlurmJobRecord)]
    assert names[0] is not None
    assert names[1:] == ['s_retry', 's_wbsync']


def test_auto_retries_only_interrupted_runs_with_progress_under_the_cap(
    project: Path, fake: FakeSlurm, caplog: pytest.LogCaptureFixture
) -> None:
    submit_study(project=project)
    enable_retry(project=project)
    # a: one more done instance than its snapshot
    folder_a = project / 'runs' / 's' / 'g/small/1'
    folder_a.mkdir(parents=True)
    write_status(
        path=folder_a / 'status.json',
        snapshot=RunStatusSnapshot(
            run_state=RunState.RUNNING,
            updated_at=utc_now_iso(),
            instances={'train': InstanceRecord(state=InstanceState.DONE)},
        ),
    )
    # b: a checkpoint newer than its snapshot
    folder_b = project / 'runs' / 's' / 'g/small/2'
    folder_b.mkdir(parents=True)
    (folder_b / 'last.ckpt').write_text('ckpt')
    # d: progress would count, but it already used all three attempts
    folder_d = project / 'runs' / 's' / 'g/large/2'
    folder_d.mkdir(parents=True)
    write_status(
        path=folder_d / 'status.json',
        snapshot=RunStatusSnapshot(
            run_state=RunState.RUNNING,
            updated_at=utc_now_iso(),
            instances={'train': InstanceRecord(state=InstanceState.DONE)},
        ),
    )
    bump_attempts(project=project, run_path='g/large/2', index=4, up_to=3)
    fake.set_answers('squeue', [{'exit': 0, 'stdout': ''}])
    fake.set_answers(
        'sacct',
        [
            {
                'exit': 0,
                'stdout': (
                    '1000_0|TIMEOUT\n1000_1|TIMEOUT\n1000_2|TIMEOUT\n2003_4|TIMEOUT\n'
                    '1000_3|FAILED\n1000_5|COMPLETED\n'
                ),
            }
        ],
    )
    with caplog.at_level('INFO'):
        code = resubmit(
            context=context(project=project, machine_file='cluster.toml'), auto=True, sleep=no_sleep
        )
    assert code == 0
    retried = [job for job in jobs_of(project=project) if job.tasks][-1]
    assert [t.run_path.as_posix() for t in retried.tasks] == ['g/small/1', 'g/small/2']
    assert all(t.attempt == 2 for t in retried.tasks)
    assert 'stuck: g/small/3 (attempt 1, no progress)' in caplog.text
    assert 'gave up: g/large/2 after 3 attempts' in caplog.text
    # the retried array and its own retry follow-up
    assert len(fake.calls('sbatch')) == 3


def _write_progress(project: Path) -> None:
    folder = project / 'runs' / 's' / 'g/large/2'
    folder.mkdir(parents=True)
    write_status(
        path=folder / 'status.json',
        snapshot=RunStatusSnapshot(
            run_state=RunState.RUNNING,
            updated_at=utc_now_iso(),
            instances={'train': InstanceRecord(state=InstanceState.DONE)},
        ),
    )


def _resubmit_large_2(  # noqa: PLR0913 # why: keyword-only test helper, one argument per scenario knob
    *,
    project: Path,
    fake: FakeSlurm,
    up_to: int,
    uncounted: tuple[int, int],
    job_state: str,
    with_progress: bool,
) -> list[TaskRecord]:
    submit_study(project=project)
    enable_retry(project=project)
    if with_progress:
        _write_progress(project)
    bump_attempts(project=project, run_path='g/large/2', index=4, up_to=up_to, uncounted=uncounted)
    fake.set_answers('squeue', [{'exit': 0, 'stdout': ''}])
    fake.set_answers('sacct', [{'exit': 0, 'stdout': f'200{up_to}_4|{job_state}\n'}])
    code = resubmit(
        context=context(project=project, machine_file='cluster.toml'), auto=True, sleep=no_sleep
    )
    assert code == 0
    last = [job for job in jobs_of(project=project) if job.tasks][-1]
    return [t for t in last.tasks if t.attempt > up_to]


def test_auto_retries_a_preempted_run_past_the_cap_without_progress(
    project: Path, fake: FakeSlurm
) -> None:
    [retried] = _resubmit_large_2(
        project=project,
        fake=fake,
        up_to=3,
        uncounted=(0, 0),
        job_state='PREEMPTED',
        with_progress=False,
    )
    assert (retried.attempt, retried.uncounted_attempts, retried.uncounted_in_row) == (4, 1, 1)


def test_preempted_attempts_do_not_count_toward_the_cap(project: Path, fake: FakeSlurm) -> None:
    [retried] = _resubmit_large_2(
        project=project,
        fake=fake,
        up_to=3,
        uncounted=(1, 1),
        job_state='TIMEOUT',
        with_progress=True,
    )
    assert retried.attempt == 4


def test_auto_gives_up_after_20_preemptions_in_a_row(
    project: Path, fake: FakeSlurm, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level('INFO'):
        retried = _resubmit_large_2(
            project=project,
            fake=fake,
            up_to=2,
            uncounted=(20, 20),
            job_state='PREEMPTED',
            with_progress=False,
        )
    assert retried == []
    assert 'in a row' in caplog.text


def test_a_counted_attempt_resets_the_in_row_counter(project: Path, fake: FakeSlurm) -> None:
    [retried] = _resubmit_large_2(
        project=project,
        fake=fake,
        up_to=2,
        uncounted=(3, 3),
        job_state='TIMEOUT',
        with_progress=True,
    )
    assert (retried.uncounted_attempts, retried.uncounted_in_row) == (3, 0)


def test_followup_chain_stops_when_nothing_is_resubmitted(project: Path, fake: FakeSlurm) -> None:
    with_followups(project=project)
    submit_study(project=project)
    submitted = len(fake.calls('sbatch'))
    fake.set_answers('sacct', [{'exit': 0, 'stdout': ''}])
    code = resubmit(
        context=context(project=project, machine_file='cluster.toml'), auto=True, sleep=no_sleep
    )
    assert code == 0
    assert len(fake.calls('sbatch')) == submitted
    assert [job.kind for job in jobs_of(project=project)] == ['run', 'followup', 'followup']


@pytest.mark.slow
def test_no_followups_locally_or_on_dry_run(project: Path, fake: FakeSlurm) -> None:
    with_followups(project=project)
    code = submit(
        context=context(project=project, machine_file='cluster.toml'),
        include_done=False,
        dry_run=True,
        extra_args=(),
        sleep=no_sleep,
    )
    assert code == 0
    assert fake.calls('sbatch') == []
    assert [job.kind for job in jobs_of(project=project)] == []
    submit_study(project=project, machine_file='local.toml')
    assert fake.calls('sbatch') == []
    assert [job.kind for job in jobs_of(project=project)] == ['run']


def test_followup_sbatch_failure_exits_one_and_skips_the_rest(
    project: Path, fake: FakeSlurm, caplog: pytest.LogCaptureFixture
) -> None:
    with_followups(project=project)
    fake.set_answers(
        'sbatch',
        [{'exit': 0, 'stdout': '1000;tslab\n'}, {'exit': 1, 'stderr': 'sbatch: error: bad'}],
    )
    with caplog.at_level('ERROR'):
        code = submit(
            context=context(project=project, machine_file='cluster.toml'),
            include_done=False,
            dry_run=False,
            extra_args=(),
            sleep=no_sleep,
        )
    assert code == 1
    assert [job.kind for job in jobs_of(project=project)] == ['run']
    assert 'follow-up s_retry not submitted' in caplog.text
    assert len(fake.calls('sbatch')) == 2


def test_retry_followup_needs_a_study_reference(
    project: Path, fake: FakeSlurm, caplog: pytest.LogCaptureFixture
) -> None:
    with_followups(project=project)
    ctx = dataclasses.replace(context(project=project, machine_file='cluster.toml'), reference='')
    with caplog.at_level('WARNING'):
        code, records = queue_followups(
            context=ctx, job_ids=['1000'], now=utc_now_iso, sleep=no_sleep
        )
    assert code == 0
    assert len(records) == 1
    assert 'automatic retry follow-up is skipped' in caplog.text


def test_followup_gets_the_absolute_path_of_a_machine_resolved_by_name(
    project: Path, fake: FakeSlurm, monkeypatch: pytest.MonkeyPatch
) -> None:
    with_followups(project=project)
    register = project / 'register.toml'
    monkeypatch.setenv('TSLAB_MACHINES_FILE', str(register))
    write_register(entries={'cluster': project / 'cluster.toml'}, environment=os.environ)
    resolved = resolve_machine(machine='cluster', start=project, environment=os.environ).path
    submit_study(project=project, machine_file=resolved.name)
    retry = jobs_of(project=project)[1]
    tasks = job_file_of(project=project, job_file=retry.job_file, suffix='tasks').read_text()
    assert f'--machine {resolved.as_posix()} --auto' in tasks
    assert resolved.is_absolute()


def test_wandb_sync_followup_warns_without_wandb(
    project: Path,
    fake: FakeSlurm,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with_followups(project=project)
    present = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        'find_spec',
        lambda name, *args: None if name == 'wandb' else present(name, *args),
    )
    code = submit(
        context=context(project=project, machine_file='cluster.toml'),
        include_done=False,
        dry_run=False,
        extra_args=(),
        sleep=no_sleep,
    )
    assert code == 0
    assert 'tslab[wandb]' in caplog.text
    assert fake.calls('sbatch') != []

"""SLURM study launch tests."""

import dataclasses
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from tests.lab.unit.study.configs import grid
from tests.lab.unit.study.helpers import context, jobs_of, no_sleep
from tests.support.fake_slurm import FakeSlurm, install_fake_slurm
from tslab.lab.enums import AfterCondition, LauncherKind, Satisfy
from tslab.lab.study import Study
from tslab.lab.study.commands import submit
from tslab.lab.study.definition import expand_runs
from tslab.lab.study.launchers.common.launch import launch as _launch
from tslab.lab.study.launchers.common.types import LaunchError, LaunchReport
from tslab.lab.study.launchers.slurm.backend import SlurmBackend
from tslab.lab.study.launchers.slurm.records import SlurmJobRecord
from tslab.lab.study.launchers.slurm.submit import _log_pattern, SlurmLauncher, write_job_files
from tslab.lab.study.machine import MachineOverride, MachineProfile, SlurmSettings
from tslab.lab.study.paths import study_folder
from tslab.lab.study.plan import assign_indices, JobGroup, plan_jobs, resolve_runs
from tslab.lab.study.plan_after import AfterTarget
from tslab.lab.study.records import read_jobs


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeSlurm:
    return install_fake_slurm(tmp_path=tmp_path, monkeypatch=monkeypatch, answers={})


@pytest.fixture
def machine(tmp_path: Path) -> MachineProfile:
    return MachineProfile(
        path=tmp_path / 'machine.toml',
        launcher=LauncherKind.SLURM,
        output_root=tmp_path / 'out',
        slurm=SlurmSettings(partition='gpu', time='01:00:00'),
        overrides=(MachineOverride(match={'recipe': ('cost-paper',)}, slurm={'gpus': 1}),),
    )


@pytest.fixture
def study() -> Study:
    return Study(name='s', grids={'g': grid(short_name='g1')})


def groups_of(study: Study, machine: MachineProfile) -> list[JobGroup]:
    runs = expand_runs(study=study)
    resolved = resolve_runs(study=study, machine=machine, runs=runs, local_cpus=1)
    return plan_jobs(
        study=study,
        machine=machine,
        resolved=resolved,
        indices=assign_indices(study=study, runs=runs, jobs=[]),
        extra_args={},
        attempts={},
        first_job_file=3,
        environment={},
    )


def launch(  # noqa: PLR0913 # why: test adapter mirrors backend launch inputs
    *,
    launcher: SlurmLauncher,
    machine: MachineProfile,
    study: Study,
    groups: list[JobGroup],
    project_root: Path,
    dry_run: bool,
) -> LaunchReport:
    return _launch(
        launcher=launcher,
        record_type=SlurmJobRecord,
        write_job_files=write_job_files,
        machine=machine,
        study=study,
        groups=groups,
        project_root=project_root,
        dry_run=dry_run,
        now=lambda: '2026-01-02T03:04:05+00:00',
    )


def test_slurm_launch_writes_submits_and_records(
    fake: FakeSlurm, machine: MachineProfile, study: Study, tmp_path: Path
) -> None:
    groups = groups_of(study, machine)
    fake.set_answers(
        command='sbatch', answers=[{'exit': 0, 'stdout': '1000'}, {'exit': 0, 'stdout': '1001'}]
    )
    folder = study_folder(machine=machine, study=study)
    report = launch(
        launcher=SlurmLauncher(sleep=lambda _: None),
        machine=machine,
        study=study,
        groups=groups,
        project_root=tmp_path,
        dry_run=False,
    )
    assert [r.job_id for r in report.submitted] == ['1000', '1001']
    jobs = read_jobs(path=folder / 'submissions.jsonl')
    assert jobs[1].tasks[0].resources['gpus'] == 1
    assert jobs[0].kind == 'run'
    assert (folder / 'jobs' / '3.tasks').read_text().splitlines()[1].startswith('1 1 ')
    assert fake.calls('sbatch')[0] == ['--parsable', str(folder / 'jobs' / '3.sbatch')]


def test_slurm_launch_resolves_same_launch_dependency(
    fake: FakeSlurm, machine: MachineProfile, study: Study, tmp_path: Path
) -> None:
    upstream, downstream = groups_of(study, machine)
    upstream_path = upstream.tasks[0].resolved.run.run_path
    downstream = dataclasses.replace(
        downstream,
        after=(
            AfterTarget(
                condition=AfterCondition.OK,
                delay_minutes=0,
                satisfy=Satisfy.ALL,
                job_tasks=(),
                run_paths=(upstream_path,),
            ),
        ),
    )
    fake.set_answers(
        command='sbatch', answers=[{'exit': 0, 'stdout': '1000'}, {'exit': 0, 'stdout': '1001'}]
    )

    report = launch(
        launcher=SlurmLauncher(sleep=lambda _: None),
        machine=machine,
        study=study,
        groups=[upstream, downstream],
        project_root=tmp_path,
        dry_run=False,
    )

    assert [record.job_id for record in report.submitted] == ['1000', '1001']
    assert fake.calls('sbatch')[1][1] == f'--dependency=afterok:1000_{upstream.tasks[0].index}'


def test_backend_writes_chain_exit_check_for_chained_grid(
    fake: FakeSlurm, project: Path, machine: MachineProfile, study: Study, tmp_path: Path
) -> None:
    ctx = context(project=project, machine_file='cluster.toml')
    group = dataclasses.replace(groups_of(study, machine)[0], chained=True)

    report = SlurmBackend().launch(
        context=ctx,
        groups=[group],
        folder=study_folder(machine=ctx.machine, study=ctx.study),
        dry_run=True,
        start_delay_seconds=None,
        sleep=no_sleep,
        task_stream=None,
        environment={},
        pid=1,
        host='test-host',
        now=lambda: '2026-01-02T03:04:05+00:00',
    )

    assert report is not None
    assert fake.calls('sbatch') == []
    script = (
        study_folder(machine=ctx.machine, study=ctx.study) / 'jobs' / f'{group.job_file}.sbatch'
    ).read_text()
    chain_command = shlex.join(
        [
            *ctx.machine.tslab_command,
            'study',
            'chain-exit',
            ctx.reference,
            '--machine',
            ctx.machine_path.as_posix(),
            '--job-file',
            str(group.job_file),
        ]
    )
    assert '#SBATCH --kill-on-invalid-dep=yes' in script
    assert f'{chain_command} --index ${{SLURM_ARRAY_TASK_ID}}' in script
    assert script.rstrip().endswith('exit "$CHECK_EXIT"')


def test_dry_run_writes_files_but_submits_nothing(
    fake: FakeSlurm, machine: MachineProfile, study: Study, tmp_path: Path
) -> None:
    groups = groups_of(study, machine)
    folder = study_folder(machine=machine, study=study)
    report = launch(
        launcher=SlurmLauncher(sleep=lambda _: None),
        machine=machine,
        study=study,
        groups=groups,
        project_root=tmp_path,
        dry_run=True,
    )
    assert report.submitted == ()
    assert fake.calls('sbatch') == []
    assert not (folder / 'submissions.jsonl').exists()
    assert (folder / 'jobs' / '3.sbatch').is_file()
    assert (folder / 'logs').is_dir()


def test_launch_reports_partial_submission(
    fake: FakeSlurm, machine: MachineProfile, study: Study, tmp_path: Path
) -> None:
    fake.set_answers(
        'sbatch', [{'exit': 0, 'stdout': '1000'}, {'exit': 1, 'stderr': 'error: invalid partition'}]
    )
    groups = groups_of(study, machine)
    folder = study_folder(machine=machine, study=study)
    report = launch(
        launcher=SlurmLauncher(sleep=lambda _: None),
        machine=machine,
        study=study,
        groups=groups,
        project_root=tmp_path,
        dry_run=False,
    )
    assert len(report.submitted) == 1
    assert len(report.not_submitted) == 1
    assert report.error is not None
    assert 'invalid partition' in report.error
    assert len(read_jobs(path=folder / 'submissions.jsonl')) == 1


def test_launcher_submit_raises_launch_error_when_sbatch_fails(
    fake: FakeSlurm, machine: MachineProfile, study: Study, tmp_path: Path
) -> None:
    fake.set_answers('sbatch', [{'exit': 1, 'stderr': 'error: invalid partition'}])
    groups = groups_of(study, machine)
    folder = study_folder(machine=machine, study=study)
    write_job_files(study_folder=folder, groups=groups, machine=machine, project_root=tmp_path)
    with pytest.raises(LaunchError, match='invalid partition'):
        SlurmLauncher(sleep=lambda _: None).submit(
            group=groups[0], study_folder=folder, record=lambda **_kwargs: None, upstream={}
        )


def test_slurm_submit_drops_code_snapshot_variables(
    machine: MachineProfile, study: Study, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('TSLAB_CODE_SNAPSHOT', '/s')
    monkeypatch.setenv('PYTHONPATH', '/s/src')
    seen: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen.update(kwargs)
        return subprocess.CompletedProcess(command, 0, stdout='77;c\n', stderr='')

    monkeypatch.setattr(subprocess, 'run', fake_run)
    group = groups_of(study, machine)[0]
    job_id = SlurmLauncher(sleep=lambda _seconds: None).submit(
        group=group, study_folder=tmp_path, record=lambda **_kwargs: None, upstream={}
    )
    assert job_id == '77'
    env = seen['env']
    assert isinstance(env, dict)
    assert 'TSLAB_CODE_SNAPSHOT' not in env
    assert 'PYTHONPATH' not in env


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


def test_resubmitted_task_keeps_its_array_index(project: Path, fake: FakeSlurm) -> None:
    submit(
        context=context(project=project, machine_file='cluster.toml'),
        include_done=False,
        dry_run=False,
        extra_args=(),
        sleep=no_sleep,
    )
    fake.set_answers('squeue', [{'exit': 0, 'stdout': '1000_0|PENDING\n1000_1|RUNNING\n'}])
    fake.set_answers('sacct', [{'exit': 0, 'stdout': '1000_2|COMPLETED\n1000_3|TIMEOUT\n'}])
    run_toy(project=project, model='large', seed=1)
    submit(
        context=context(project=project, machine_file='cluster.toml'),
        include_done=False,
        dry_run=False,
        extra_args=(),
        sleep=no_sleep,
    )
    jobs = jobs_of(project=project)
    script = (project / 'runs' / 's' / '.study' / 'jobs' / f'{jobs[1].job_file}.sbatch').read_text()
    assert '#SBATCH --array=3%100' in script
    assert '#SBATCH --job-name=s_g1' in script
    assert 'logs/s_g1_%a.out' in script


def test_dry_run_writes_the_script_and_describes_resources(
    project: Path, fake: FakeSlurm, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level('INFO'):
        submit(
            context=context(project=project, machine_file='cluster.toml'),
            include_done=False,
            dry_run=True,
            extra_args=(),
            sleep=no_sleep,
        )
    assert (project / 'runs' / 's' / '.study' / 'jobs' / '1.sbatch').is_file()
    text = ' '.join(caplog.text.split())
    assert 'g1 4 partition=' in text


def test_write_job_files_writes_the_start_delay(
    machine: MachineProfile, study: Study, tmp_path: Path
) -> None:
    folder = study_folder(machine=machine, study=study)
    write_job_files(
        study_folder=folder,
        groups=groups_of(study, machine),
        machine=machine,
        project_root=tmp_path,
        start_delay_seconds=660,
    )
    script = (folder / 'jobs' / '3.sbatch').read_text()
    assert '#SBATCH --begin=now+660' in script


def test_job_name_names_the_job_and_its_log_pattern(
    machine: MachineProfile, tmp_path: Path
) -> None:
    study = Study(name='s', grids={'g': grid(short_name='g1', job_name='{recipe}_{short_name}')})
    groups = groups_of(study, machine)
    folder = study_folder(machine=machine, study=study)
    write_job_files(study_folder=folder, groups=groups, machine=machine, project_root=tmp_path)
    scripts = [(folder / 'jobs' / f'{group.job_file}.sbatch').read_text() for group in groups]
    assert any('--job-name=cost-paper_g1' in text for text in scripts)
    for group, text in zip(groups, scripts, strict=True):
        assert f'--output={folder}/logs/{group.job_name}_%a.out' in text


def test_log_pattern_refuses_a_log_name_it_cannot_read(
    machine: MachineProfile, study: Study
) -> None:
    group = groups_of(study, machine)[0]
    task = dataclasses.replace(group.tasks[0], log_file=Path('logs') / 'odd-name.log')
    broken = dataclasses.replace(group, tasks=(task, *group.tasks[1:]))
    with pytest.raises(ValueError, match=r'odd-name\.log'):
        _log_pattern(group=broken)

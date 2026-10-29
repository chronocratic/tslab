"""Launching study jobs on SLURM or locally: files, submission, records."""

from io import StringIO
from pathlib import Path
import sys

import pytest

from tests.unit.study.configs import grid
from tests.unit.study.fake_slurm import FakeSlurm, install_fake_slurm
from tslab.study import Study
from tslab.study.definition import expand_runs
from tslab.study.launch import launch, LocalLauncher, SlurmLauncher
from tslab.study.machine import MachineOverride, MachineProfile, SlurmSettings
from tslab.study.plan import assign_indices, JobGroup, plan_jobs, resolve_runs, study_folder
from tslab.study.records import read_jobs


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeSlurm:
    return install_fake_slurm(tmp_path=tmp_path, monkeypatch=monkeypatch, answers={})


@pytest.fixture
def machine(tmp_path: Path) -> MachineProfile:
    return MachineProfile(
        path=tmp_path / 'machine.toml',
        launcher='slurm',
        output_root=tmp_path / 'out',
        slurm=SlurmSettings(partition='gpu', time='01:00:00'),
        overrides=(MachineOverride(match={'recipe': ('cost-paper',)}, slurm={'gpus': 1}),),
    )


@pytest.fixture
def study() -> Study:
    return Study(name='s', grids={'g': grid(short_name='g1')})


def groups_of(study: Study, machine: MachineProfile) -> list[JobGroup]:
    runs = expand_runs(study=study)
    resolved = resolve_runs(study=study, machine=machine, runs=runs)
    return plan_jobs(
        study=study,
        machine=machine,
        resolved=resolved,
        indices=assign_indices(runs=runs, jobs=[]),
        extra_args={},
        attempts={},
        first_job_file=3,
    )


def test_slurm_launch_writes_submits_and_records(
    fake: FakeSlurm, machine: MachineProfile, study: Study, tmp_path: Path
) -> None:
    groups = groups_of(study, machine)
    fake.set_answers('sbatch', [{'exit': 0, 'stdout': '1000'}, {'exit': 0, 'stdout': '1001'}])
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


def test_local_launcher_runs_in_order_and_continues_after_failure(
    machine: MachineProfile, tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    machine = MachineProfile(
        path=machine.path, launcher='local', output_root=machine.output_root, slurm=None
    )
    script = tmp_path / 'run.py'
    script.write_text(
        'import sys\n'
        'args = sys.argv[1:]\n'
        "print(' '.join(args))\n"
        "if '--run-path' in args and args[args.index('--run-path') + 1].endswith('/7'):\n"
        '    sys.exit(1)\n'
    )
    study = Study(
        name='s', grids={'g': grid(short_name='g1', command=(sys.executable, str(script)))}
    )
    runs = expand_runs(study=study)
    resolved = resolve_runs(study=study, machine=machine, runs=runs)
    groups = plan_jobs(
        study=study,
        machine=machine,
        resolved=resolved,
        indices=assign_indices(runs=runs, jobs=[]),
        extra_args={},
        attempts={},
        first_job_file=1,
    )
    assert len(groups) == 1
    report = launch(
        launcher=LocalLauncher(project_root=tmp_path, env={}),
        machine=machine,
        study=study,
        groups=groups,
        project_root=tmp_path,
        dry_run=False,
    )
    assert report.submitted[0].job_id.startswith('local-')
    assert len(report.submitted[0].tasks) == 4
    folder = study_folder(machine=machine, study=study)
    log = folder / 'logs' / 'g1_2.out'
    assert log.read_text().startswith('=== tslab attempt 1 | job local-')
    console = capfd.readouterr().out
    assert '--run-path' in console
    # launching the same tasks again (attempt 2) appends: the log holds both headers
    attempts = {run.run_path: 1 for run in runs}
    groups2 = plan_jobs(
        study=study,
        machine=machine,
        resolved=resolved,
        indices=assign_indices(runs=runs, jobs=[]),
        extra_args={},
        attempts=attempts,
        first_job_file=2,
    )
    launch(
        launcher=LocalLauncher(project_root=tmp_path, env={}),
        machine=machine,
        study=study,
        groups=groups2,
        project_root=tmp_path,
        dry_run=False,
    )
    assert log.read_text().count('=== tslab attempt') == 2


def test_local_launcher_writes_to_given_stream(machine: MachineProfile, tmp_path: Path) -> None:
    local_machine = MachineProfile(
        path=machine.path, launcher='local', output_root=machine.output_root, slurm=None
    )
    script = tmp_path / 'echo.py'
    script.write_text("print('TASK-OUTPUT')")
    study = Study(
        name='s', grids={'g': grid(short_name='g1', command=(sys.executable, str(script)))}
    )
    stream = StringIO()
    launch(
        launcher=LocalLauncher(project_root=tmp_path, env={}, stream=stream),
        machine=local_machine,
        study=study,
        groups=groups_of(study, local_machine),
        project_root=tmp_path,
        dry_run=False,
    )
    assert 'TASK-OUTPUT' in stream.getvalue()


def test_local_launch_records_the_job_before_tasks_run_and_survives_a_missing_command(
    machine: MachineProfile, tmp_path: Path
) -> None:
    study = Study(
        name='s', grids={'g': grid(short_name='g1', command=(str(tmp_path / 'no-such-command'),))}
    )
    runs = expand_runs(study=study)
    groups = plan_jobs(
        study=study,
        machine=machine,
        resolved=resolve_runs(study=study, machine=machine, runs=runs),
        indices=assign_indices(runs=runs, jobs=[]),
        extra_args={},
        attempts={},
        first_job_file=1,
    )
    report = launch(
        launcher=LocalLauncher(project_root=tmp_path, env={}),
        machine=machine,
        study=study,
        groups=groups,
        project_root=tmp_path,
        dry_run=False,
    )
    folder = study_folder(machine=machine, study=study)
    assert len(report.submitted) == len(groups)
    assert len(read_jobs(path=folder / 'submissions.jsonl')) == len(groups)
    logs = ''.join(path.read_text() for path in (folder / 'logs').glob('*.out'))
    assert logs.count('could not start') == sum(len(group.tasks) for group in groups)

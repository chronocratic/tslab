"""SLURM study launch tests."""

from pathlib import Path

import pytest

from tests.support.fake_slurm import FakeSlurm, install_fake_slurm
from tests.unit.study.configs import grid
from tslab.study import Study
from tslab.study.definition import expand_runs
from tslab.study.launchers import launch as _launch, LaunchReport
from tslab.study.launchers.slurm.submit import SlurmLauncher, write_job_files
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
    resolved = resolve_runs(study=study, machine=machine, runs=runs, local_cpus=1)
    return plan_jobs(
        study=study,
        machine=machine,
        resolved=resolved,
        indices=assign_indices(runs=runs, jobs=[]),
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

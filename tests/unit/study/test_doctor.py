"""Machine and project readiness checks."""

from pathlib import Path

import pytest

from tests.unit.study.fake_slurm import FakeSlurm
from tslab.study.doctor import run_checks


def test_doctor_ok_on_local_toy_profile(project: Path) -> None:
    checks = run_checks(machine_path=project / 'local.toml', study=None, start=project)
    statuses = {check.name: check.status for check in checks}
    assert statuses['profile'] == 'ok'
    assert statuses['output_root'] == 'ok'
    assert statuses['disk'] == 'ok'


def test_doctor_fails_missing_slurm_commands(
    project: Path, fake: FakeSlurm, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    empty_path = tmp_path / 'empty-bin'
    empty_path.mkdir()
    monkeypatch.setenv('PATH', str(empty_path))
    checks = run_checks(machine_path=project / 'cluster.toml', study=None, start=project)
    slurm = next(check for check in checks if check.name == 'slurm')
    assert slurm.status == 'fail'
    assert 'sbatch' in slurm.message
    assert 'squeue' in slurm.message
    assert 'sacct' in slurm.message


def test_doctor_fails_unset_env_var(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('TSLAB_DOCTOR_MISSING', raising=False)
    profile = project / 'unset.toml'
    profile.write_text(
        (project / 'local.toml').read_text(encoding='utf-8')
        + '\n[env]\nDOCTOR_VALUE = "${TSLAB_DOCTOR_MISSING}"\n',
        encoding='utf-8',
    )
    checks = run_checks(machine_path=profile, study=None, start=project)
    assert checks[0].name == 'profile'
    assert checks[0].status == 'fail'
    assert 'TSLAB_DOCTOR_MISSING' in checks[0].message
    assert 'output_root' not in {check.name for check in checks}


def test_doctor_warns_low_disk(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checks = run_checks(
        machine_path=project / 'local.toml', study=None, start=project, free_bytes=lambda _: 1
    )
    disk = next(check for check in checks if check.name == 'disk')
    assert disk.status == 'warn'
    assert '0.0 GB' in disk.message


def test_doctor_checks_study_fits_machine(project: Path, fake: FakeSlurm) -> None:
    checks = run_checks(
        machine_path=project / 'cluster.toml', study='toy_studies:study', start=project
    )
    assert next(check for check in checks if check.name == 'study').status == 'ok'

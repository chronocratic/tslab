"""Machine and project readiness checks."""

import os
from pathlib import Path

import pytest

from tests.support.fake_backend import FakeBackend
from tslab.core.enums import CheckStatus
import tslab.study.doctor as doctor_module
from tslab.study.doctor import DoctorCheck, run_checks


def test_doctor_ok_on_local_toy_profile(project: Path) -> None:
    checks = run_checks(
        machine_path=project / 'local.toml', study=None, start=project, environment=os.environ
    )
    statuses = {check.name: check.status for check in checks}
    assert statuses['profile'] == 'ok'
    assert statuses['output_root'] == 'ok'
    assert statuses['disk'] == 'ok'


def test_doctor_checks_come_from_selected_backend(project: Path, fake_backend: FakeBackend) -> None:
    backend_check = DoctorCheck(name='backend', status=CheckStatus.OK, message='backend selected')
    fake_backend.doctor_answer = (backend_check,)

    checks = run_checks(
        machine_path=project / 'local.toml', study=None, start=project, environment=os.environ
    )

    assert backend_check in checks
    assert len(fake_backend.calls('doctor_checks')) == 1


def test_wandb_check_uses_given_environment(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Netrc:
        def authenticators(self, _host: str) -> None:
            return None

    monkeypatch.setattr(doctor_module.netrc, 'netrc', Netrc)

    checks = run_checks(
        machine_path=project / 'local.toml',
        study=None,
        start=project,
        environment={'WANDB_API_KEY': 'token'},
    )

    wandb = next(check for check in checks if check.name == 'wandb')
    assert wandb.status == 'ok'


def test_doctor_fails_unset_env_var(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('TSLAB_DOCTOR_MISSING', raising=False)
    profile = project / 'unset.toml'
    profile.write_text(
        (project / 'local.toml').read_text(encoding='utf-8')
        + '\n[env]\nDOCTOR_VALUE = "${TSLAB_DOCTOR_MISSING}"\n',
        encoding='utf-8',
    )
    checks = run_checks(machine_path=profile, study=None, start=project, environment=os.environ)
    assert checks[0].name == 'profile'
    assert checks[0].status == 'fail'
    assert 'TSLAB_DOCTOR_MISSING' in checks[0].message
    assert 'output_root' not in {check.name for check in checks}


def test_doctor_warns_low_disk(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checks = run_checks(
        machine_path=project / 'local.toml',
        study=None,
        start=project,
        environment=os.environ,
        free_bytes=lambda _: 1,
    )
    disk = next(check for check in checks if check.name == 'disk')
    assert disk.status == 'warn'
    assert '0.0 GB' in disk.message


def test_doctor_checks_study_fits_machine(project: Path, fake_backend: FakeBackend) -> None:
    checks = run_checks(
        machine_path=project / 'cluster.toml',
        study='toy_studies:study',
        start=project,
        environment=os.environ,
    )
    assert next(check for check in checks if check.name == 'study').status == 'ok'

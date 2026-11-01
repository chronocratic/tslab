"""SLURM command readiness checks."""

import os
from pathlib import Path

import pytest

from tests.support.fake_slurm import FakeSlurm
from tslab.study.doctor import run_checks
from tslab.study.launchers.slurm import doctor


def test_slurm_check_lists_missing_scheduler_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    def which(name: str) -> str | None:
        return None if name == 'sacct' else '/bin/' + name

    monkeypatch.setattr(doctor.shutil, 'which', which)

    check = doctor.slurm_check()

    assert (check.name, check.status) == ('slurm', 'fail')
    assert check.message == 'missing SLURM commands: sacct'


def test_doctor_fails_missing_slurm_commands(
    project: Path, fake: FakeSlurm, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    empty_path = tmp_path / 'empty-bin'
    empty_path.mkdir()
    monkeypatch.setenv('PATH', str(empty_path))
    checks = run_checks(
        machine_path=project / 'cluster.toml', study=None, start=project, environment=os.environ
    )
    slurm = next(check for check in checks if check.name == 'slurm')
    assert slurm.status == 'fail'
    assert 'sbatch' in slurm.message
    assert 'squeue' in slurm.message
    assert 'sacct' in slurm.message

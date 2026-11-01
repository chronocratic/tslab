"""SLURM command readiness checks."""

import pytest

from tslab.study.launchers.slurm import doctor


def test_slurm_check_lists_missing_scheduler_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    def which(name: str) -> str | None:
        return None if name == 'sacct' else '/bin/' + name

    monkeypatch.setattr(doctor.shutil, 'which', which)

    check = doctor.slurm_check()

    assert (check.name, check.status) == ('slurm', 'fail')
    assert check.message == 'missing SLURM commands: sacct'

"""Fixtures shared by the launcher tests: fake SLURM commands on PATH."""

from pathlib import Path

import pytest

from tests.support.fake_slurm import FakeSlurm, install_fake_slurm


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeSlurm:
    """Fake SLURM commands first on PATH."""
    return install_fake_slurm(tmp_path=tmp_path, monkeypatch=monkeypatch, answers={})

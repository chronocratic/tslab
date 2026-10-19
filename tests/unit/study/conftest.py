"""Fixtures shared by the study tests: a toy project, fake SLURM, clean imports."""

from collections.abc import Iterator
from pathlib import Path
import sys

import pytest

from tests.unit.study.fake_slurm import FakeSlurm, install_fake_slurm
from tests.unit.study.project import write_toy_project

_TOY_MODULES = ('toy_studies', 'toy_script')


@pytest.fixture(autouse=True)
def _clean_study_imports() -> Iterator[None]:
    """Drop the toy modules and the project folder from the import state after a test."""
    path = list(sys.path)
    yield
    for name in _TOY_MODULES:
        sys.modules.pop(name, None)
    sys.path[:] = path


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A toy project folder; the test runs from inside it."""
    folder = write_toy_project(project=tmp_path / 'proj')
    # why: load_study imports from the current directory, like a user in a project
    monkeypatch.chdir(folder)
    return folder


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeSlurm:
    """Fake SLURM commands first on PATH."""
    return install_fake_slurm(tmp_path=tmp_path, monkeypatch=monkeypatch, answers={})

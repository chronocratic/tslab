"""Fixtures shared by the study tests: a toy project, a fake backend, clean imports."""

from collections.abc import Iterator
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from tests.support.fake_backend import FakeBackend, install_fake_backend
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


@pytest.fixture(autouse=True)
def _no_grace_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the wait for a late ``status.json`` instant; only ``reporting.state`` sees it."""
    monkeypatch.setattr(
        'tslab.study.reporting.state.time', SimpleNamespace(sleep=lambda _seconds: None)
    )


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A toy project folder; the test runs from inside it."""
    folder = write_toy_project(project=tmp_path / 'proj')
    # why: load_study imports from the current directory, like a user in a project
    monkeypatch.chdir(folder)
    return folder


@pytest.fixture
def fake_backend(monkeypatch: pytest.MonkeyPatch) -> FakeBackend:
    """The fake backend installed for both launchers."""
    return install_fake_backend(monkeypatch=monkeypatch)

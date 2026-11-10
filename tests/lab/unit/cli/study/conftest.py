"""Fixtures shared by the study command tests: a fake backend, clean toy imports."""

from collections.abc import Iterator
import sys

import pytest

from tests.support.fake_backend import FakeBackend, install_fake_backend

_TOY_MODULES = ('toy_studies', 'toy_script')


@pytest.fixture(autouse=True)
def _clean_toy_imports() -> Iterator[None]:
    path = list(sys.path)
    yield
    for name in _TOY_MODULES:
        sys.modules.pop(name, None)
    sys.path[:] = path


@pytest.fixture
def fake_backend(monkeypatch: pytest.MonkeyPatch) -> FakeBackend:
    """The fake backend installed for both launchers."""
    return install_fake_backend(monkeypatch=monkeypatch)

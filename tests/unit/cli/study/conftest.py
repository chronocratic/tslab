"""Fixtures shared by the study command tests: clean toy imports."""

from collections.abc import Iterator
import sys

import pytest

_TOY_MODULES = ('toy_studies', 'toy_script')


@pytest.fixture(autouse=True)
def _clean_toy_imports() -> Iterator[None]:
    path = list(sys.path)
    yield
    for name in _TOY_MODULES:
        sys.modules.pop(name, None)
    sys.path[:] = path

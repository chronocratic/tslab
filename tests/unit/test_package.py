from importlib import import_module
from importlib.metadata import version
import re

import pytest

import tslab

OLD_MODULES = (
    'tslab.artifact',
    'tslab.config',
    'tslab.fingerprint',
    'tslab.meta',
    'tslab.paths',
    'tslab.run',
    'tslab.seeding',
    'tslab.stage',
    'tslab.status',
)


def test_version_comes_from_the_installed_package() -> None:
    assert tslab.__version__ == version('tslab')
    assert re.fullmatch(r'\d+\.\d+\.\d+.*', tslab.__version__)


@pytest.mark.parametrize('name', OLD_MODULES)
def test_old_module_paths_are_gone(name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        import_module(name)

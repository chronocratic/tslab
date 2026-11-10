from importlib.metadata import version
import re
import subprocess
import sys

import pytest

import tslab


def test_version_comes_from_the_installed_package() -> None:
    assert tslab.__version__ == version('tslab')
    assert re.fullmatch(r'\d+\.\d+\.\d+.*', tslab.__version__)


@pytest.mark.parametrize(
    'old',
    [
        'tslab.study',
        'tslab.core',
        'tslab.jobs',
        'tslab.stages',
        'tslab.interfaces',
        'tslab.tasks',
        'tslab.probes',
        'tslab.lightning',
        'tslab.tracking',
        'tslab.adapters',
        'tslab.enums',
        'tslab.analysis',
        'tslab.cli.study',
    ],
)
def test_old_path_is_gone(old: str) -> None:
    result = subprocess.run(  # noqa: S603 # why: fresh interpreter, fixed arguments
        [sys.executable, '-c', f'import {old}'], capture_output=True, text=True, check=False
    )
    assert 'ModuleNotFoundError' in result.stderr


def test_top_level_exports_only_the_version() -> None:
    assert tslab.__all__ == ['__version__']

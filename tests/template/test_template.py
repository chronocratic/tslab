"""A project made by ``tslab new`` installs, passes its own checks and runs its smoke test.

The project is installed against this checkout of TSLab instead of the
released version its ``pyproject.toml`` asks for.
"""

import os
from pathlib import Path
import shutil
import subprocess

import pytest

from tslab.cli.new import create_project

REPOSITORY = Path(__file__).parents[2]
RELEASED_TSLAB = '>=0.1.0a1,<0.2"]'


def uv(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    uv_path = shutil.which('uv')
    assert uv_path is not None, 'uv is not installed'
    # Without this, uv would use the virtual environment of the TSLab checkout.
    environment = {key: value for key, value in os.environ.items() if key != 'VIRTUAL_ENV'}
    return subprocess.run(  # noqa: S603 # why: runs uv on the generated project
        [uv_path, *args], cwd=cwd, env=environment, capture_output=True, text=True, check=False
    )


@pytest.mark.slow
@pytest.mark.chronocratic
def test_generated_project_passes_its_checks(tmp_path: Path) -> None:
    project = tmp_path / 'my-paper'
    assert create_project(path=project) == 'my_paper'
    pyproject = project / 'pyproject.toml'
    text = pyproject.read_text(encoding='utf-8')
    assert RELEASED_TSLAB in text
    text = text.replace(RELEASED_TSLAB, '"]')
    source = f'tslab = {{ path = "{REPOSITORY.as_posix()}", editable = true }}'
    text += f'\n[tool.uv.sources]\n{source}\n'
    pyproject.write_text(text, encoding='utf-8')

    for command in (
        ('sync',),
        ('run', 'ruff', 'check', '.'),
        ('run', 'ruff', 'format', '--check', '.'),
        ('run', 'ty', 'check'),
        ('run', 'pytest', '-x', '-q'),
    ):
        result = uv(*command, cwd=project)
        assert result.returncode == 0, f'uv {" ".join(command)}:\n{result.stdout}\n{result.stderr}'

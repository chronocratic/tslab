"""A project made by ``tslab new`` installs, passes its own checks and runs its smoke test.

The project is installed against this checkout of TSLab instead of the
released version its ``pyproject.toml`` asks for.
"""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from tslab.cli import TS_TEMPLATE
from tslab.lab.cli.new import create_project, LAB_TEMPLATE
from tslab.lab.study.project import load_project_settings

REPOSITORY = Path(__file__).parents[3]
RELEASED_TSLAB = '>=0.1.0a1,<0.2"]'


def uv(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    uv_path = shutil.which('uv')
    assert uv_path is not None, 'uv is not installed'
    # Without this, uv would use the virtual environment of the TSLab checkout.
    environment = {key: value for key, value in os.environ.items() if key != 'VIRTUAL_ENV'}
    return subprocess.run(  # noqa: S603 # why: runs uv on the generated project
        [uv_path, *args], cwd=cwd, env=environment, capture_output=True, text=True, check=False
    )


def check_generated_project(*, project: Path) -> None:
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


@pytest.mark.slow
@pytest.mark.chronocratic
def test_generated_project_passes_its_checks(tmp_path: Path) -> None:
    project = tmp_path / 'my-paper'
    assert create_project(path=project, template=TS_TEMPLATE) == 'my_paper'
    check_generated_project(project=project)


@pytest.mark.slow
def test_generated_lab_project_passes_its_checks(tmp_path: Path) -> None:
    project = tmp_path / 'my-paper'
    assert create_project(path=project, template=LAB_TEMPLATE) == 'my_paper'
    check_generated_project(project=project)


def files_below(*, directory: Path) -> set[str]:
    return {
        path.relative_to(directory).as_posix() for path in directory.rglob('*') if path.is_file()
    }


def test_project_is_copied_with_names_filled_in(tmp_path: Path) -> None:
    project = tmp_path / 'my-paper'
    assert create_project(path=project, template=TS_TEMPLATE) == 'my_paper'
    names = files_below(directory=project)
    assert {'pyproject.toml', 'src/my_paper/__init__.py'} <= names
    for name in names:
        assert '{{' not in (project / name).read_text(encoding='utf-8'), name


def test_template_tslab_toml_loads_and_picks_no_machine(tmp_path: Path) -> None:
    project = tmp_path / 'my-paper'
    create_project(path=project, template=TS_TEMPLATE)
    settings = load_project_settings(start=project)
    assert settings.studies == 'my_paper.studies'
    assert settings.runs == project / 'runs'
    assert settings.machine_by_host == ()
    for name in ('README.md', 'AGENTS.md'):
        assert 'tslab study submit example' in (project / name).read_text(encoding='utf-8')


def test_template_study_status_examples_are_one_shot(tmp_path: Path) -> None:
    project = tmp_path / 'my-paper'
    create_project(path=project, template=TS_TEMPLATE)

    for name in ('README.md', 'AGENTS.md'):
        lines = (project / name).read_text(encoding='utf-8').splitlines()
        line = next(line for line in lines if line.startswith('uv run tslab study status'))
        command, _, comment = line.partition('#')
        assert '--watch 60' not in command
        assert 'add --watch 60 to refresh' in comment


def test_template_study_commands_include_agent_workflow(tmp_path: Path) -> None:
    project = tmp_path / 'my-paper'
    create_project(path=project, template=TS_TEMPLATE)

    for name in ('README.md', 'AGENTS.md'):
        content = (project / name).read_text(encoding='utf-8')
        for command in ('tslab study wait', 'tslab study failures', 'tslab study show'):
            assert command in content, (name, command)


def test_new_copies_the_agent_skill(tmp_path: Path) -> None:
    project = tmp_path / 'my-paper'
    create_project(path=project, template=TS_TEMPLATE)
    skill = (project / '.agents/skills/tslab/SKILL.md').read_text(encoding='utf-8')
    assert 'tslab study wait' in skill
    assert '{{' not in skill
    assert '.agents/skills/tslab/SKILL.md' in (project / 'AGENTS.md').read_text(encoding='utf-8')


def test_name_option_gives_the_package_name(tmp_path: Path) -> None:
    assert create_project(path=tmp_path / 'x', template=TS_TEMPLATE, name='Robust-TS') == (
        'robust_ts'
    )
    assert (tmp_path / 'x' / 'src' / 'robust_ts' / 'classify.py').exists()


def test_long_package_name_passes_ruff(tmp_path: Path) -> None:
    # A long package name lengthens the import lines of the template files.
    project = tmp_path / 'robust-forecasting-study'
    create_project(path=project, template=TS_TEMPLATE)
    for command in (('check', '.'), ('format', '--check', '.')):
        result = subprocess.run(  # noqa: S603 # why: runs this interpreter's ruff
            [sys.executable, '-m', 'ruff', *command],
            cwd=project,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f'ruff {" ".join(command)}:\n{result.stdout}'

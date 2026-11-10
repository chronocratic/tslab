from pathlib import Path
import re

import pytest

from tslab.lab.cli import main
from tslab.lab.cli.new import create_project, LAB_TEMPLATE, New
from tslab.lab.study.project import load_project_settings


def files_below(*, directory: Path) -> set[str]:
    return {
        path.relative_to(directory).as_posix() for path in directory.rglob('*') if path.is_file()
    }


def test_project_is_copied_with_names_filled_in(tmp_path: Path) -> None:
    project = tmp_path / 'my-paper'
    assert create_project(path=project, template=LAB_TEMPLATE) == 'my_paper'
    names = files_below(directory=project)
    assert {'pyproject.toml', 'src/my_paper/__init__.py', 'src/my_paper/stages.py'} <= names
    for name in names:
        assert '{{' not in (project / name).read_text(encoding='utf-8'), name
    assert 'name = "my-paper"' in (project / 'pyproject.toml').read_text(encoding='utf-8')


def test_lab_template_has_no_ml_or_ts_code(tmp_path: Path) -> None:
    create_project(path=tmp_path / 'p', template=LAB_TEMPLATE)
    text = ' '.join(f.read_text() for f in (tmp_path / 'p').rglob('*') if f.is_file())
    assert not re.search(r'(?i)torch|lightning|forecast|encoder|probe|time series', text)


def test_template_tslab_toml_loads_and_picks_no_machine(tmp_path: Path) -> None:
    project = tmp_path / 'my-paper'
    create_project(path=project, template=LAB_TEMPLATE)
    settings = load_project_settings(start=project)
    assert settings.studies == 'my_paper.studies'
    assert settings.runs == project / 'runs'
    assert settings.machine_by_host == ()
    for name in ('README.md', 'AGENTS.md'):
        assert 'tslab study submit example' in (project / name).read_text(encoding='utf-8')


def test_name_option_overrides_the_folder_name(tmp_path: Path) -> None:
    assert create_project(path=tmp_path / 'x', template=LAB_TEMPLATE, name='Robust-TS') == (
        'robust_ts'
    )
    assert (tmp_path / 'x' / 'src' / 'robust_ts' / 'stages.py').exists()


def test_empty_folder_is_accepted(tmp_path: Path) -> None:
    (tmp_path / 'p').mkdir()
    create_project(path=tmp_path / 'p', template=LAB_TEMPLATE)
    assert (tmp_path / 'p' / 'README.md').exists()


def test_existing_content_is_never_touched(tmp_path: Path) -> None:
    (tmp_path / 'p').mkdir()
    (tmp_path / 'p' / 'notes.txt').write_text('mine', encoding='utf-8')
    with pytest.raises(ValueError, match='not an empty folder'):
        create_project(path=tmp_path / 'p', template=LAB_TEMPLATE)
    assert files_below(directory=tmp_path / 'p') == {'notes.txt'}


@pytest.mark.parametrize('name', ['1paper', 'my_paper', 'my paper', ''])
def test_bad_names_are_rejected(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match='must start with a letter'):
        create_project(path=tmp_path / 'p', template=LAB_TEMPLATE, name=name)
    assert not (tmp_path / 'p').exists()


def test_string_path_is_rejected() -> None:
    with pytest.raises(TypeError, match=r'must be pathlib\.Path'):
        create_project(path='my-paper', template=LAB_TEMPLATE)  # ty: ignore[invalid-argument-type]


def test_command_line_entry_point(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as raised:
        main(args=['new', str(tmp_path / 'my-paper')])
    assert raised.value.code == 0
    assert (tmp_path / 'my-paper' / 'src' / 'my_paper' / 'stages.py').exists()
    with pytest.raises(SystemExit) as raised:
        main(args=['new', str(tmp_path / 'my-paper')])
    assert raised.value.code == 1


def test_unknown_template_is_a_usage_error(tmp_path: Path) -> None:
    assert New(path=tmp_path / 'p', template='nope').run() == 1
    assert not (tmp_path / 'p').exists()


@pytest.mark.parametrize('name', ['class', 'import', 'def'])
def test_python_keyword_names_are_rejected(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match='Python keyword'):
        create_project(path=tmp_path / 'p', template=LAB_TEMPLATE, name=name)
    assert not (tmp_path / 'p').exists()


def test_new_creates_a_lab_project_without_any_entry_point(tmp_path: Path) -> None:
    assert New(path=tmp_path / 'p').run() == 0
    assert (tmp_path / 'p' / 'src' / 'p' / 'stages.py').exists()

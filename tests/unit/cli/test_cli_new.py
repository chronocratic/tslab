from pathlib import Path

import pytest

from tslab.cli import main
from tslab.cli.new import create_project


def files_below(*, directory: Path) -> set[str]:
    return {
        path.relative_to(directory).as_posix() for path in directory.rglob('*') if path.is_file()
    }


def test_project_is_copied_with_names_filled_in(tmp_path: Path) -> None:
    project = tmp_path / 'my-paper'
    assert create_project(path=project) == 'my_paper'
    names = files_below(directory=project)
    assert {'pyproject.toml', 'src/my_paper/__init__.py'} <= names
    for name in names:
        assert '{{' not in (project / name).read_text(encoding='utf-8'), name
    assert 'name = "my-paper"' in (project / 'pyproject.toml').read_text(encoding='utf-8')


def test_name_option_overrides_the_folder_name(tmp_path: Path) -> None:
    assert create_project(path=tmp_path / 'x', name='Robust-TS') == 'robust_ts'
    assert (tmp_path / 'x' / 'src' / 'robust_ts' / 'classify.py').exists()


def test_empty_folder_is_accepted(tmp_path: Path) -> None:
    (tmp_path / 'p').mkdir()
    create_project(path=tmp_path / 'p')
    assert (tmp_path / 'p' / 'README.md').exists()


def test_existing_content_is_never_touched(tmp_path: Path) -> None:
    (tmp_path / 'p').mkdir()
    (tmp_path / 'p' / 'notes.txt').write_text('mine', encoding='utf-8')
    with pytest.raises(ValueError, match='not an empty folder'):
        create_project(path=tmp_path / 'p')
    assert files_below(directory=tmp_path / 'p') == {'notes.txt'}


@pytest.mark.parametrize('name', ['1paper', 'my_paper', 'my paper', ''])
def test_bad_names_are_rejected(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match='must start with a letter'):
        create_project(path=tmp_path / 'p', name=name)
    assert not (tmp_path / 'p').exists()


def test_string_path_is_rejected() -> None:
    with pytest.raises(TypeError, match=r'must be pathlib\.Path'):
        create_project(path='my-paper')  # ty: ignore[invalid-argument-type]


def test_command_line_entry_point(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as raised:
        main(args=['new', str(tmp_path / 'my-paper')])
    assert raised.value.code == 0
    with pytest.raises(SystemExit) as raised:
        main(args=['new', str(tmp_path / 'my-paper')])
    assert raised.value.code == 1


@pytest.mark.parametrize('name', ['class', 'import', 'def'])
def test_python_keyword_names_are_rejected(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match='Python keyword'):
        create_project(path=tmp_path / 'p', name=name)
    assert not (tmp_path / 'p').exists()

import logging
from pathlib import Path
import re
import tomllib
from typing import Any

import pytest

from tslab.lab.cli import new as new_module
from tslab.lab.cli.entrypoint import main
from tslab.lab.cli.new import create_project, LAB_TEMPLATE, Layout, New
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


def write_parent(*, folder: Path, text: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    file = folder / 'pyproject.toml'
    file.write_text(text, encoding='utf-8')
    return file


def read_toml(*, file: Path) -> dict[str, Any]:
    return tomllib.loads(file.read_text(encoding='utf-8'))


def test_no_parent_means_standalone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(new_module, '_find_parent', lambda *, path: None)  # noqa: ARG005
    create_project(path=tmp_path / 'p', template=LAB_TEMPLATE)
    data = read_toml(file=tmp_path / 'p' / 'pyproject.toml')
    assert data['project']['dependencies'] == ['tslab>=0.1.0a1,<0.2']
    assert 'sources' not in data.get('tool', {}).get('uv', {})


def test_nested_project_becomes_a_workspace_member(tmp_path: Path) -> None:
    parent = write_parent(folder=tmp_path, text='[project]\nname = "my-project"\nversion = "1"')
    create_project(path=tmp_path / 'sub' / 'exp', template=LAB_TEMPLATE)
    data = read_toml(file=parent)
    assert data['tool']['uv']['workspace'] == {'members': ['sub/exp']}
    assert data['tool']['uv']['sources'] == {'exp': {'workspace': True}}
    assert data['dependency-groups'] == {'dev': ['exp']}
    assert parent.read_text(encoding='utf-8').endswith('\n')
    member = read_toml(file=tmp_path / 'sub' / 'exp' / 'pyproject.toml')
    assert member['project']['dependencies'] == ['my-project', 'tslab>=0.1.0a1,<0.2']
    assert member['tool']['uv']['sources'] == {'my-project': {'workspace': True}}


def test_existing_parent_tables_are_left_alone(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    text = (
        '[project]\nname = "my-project"\n\n[tool.uv.sources]\nother = { workspace = true }\n\n'
        '[dependency-groups]\ndev = ["pytest"]\n'
    )
    parent = write_parent(folder=tmp_path, text=text)
    with caplog.at_level(logging.WARNING):
        create_project(path=tmp_path / 'exp', template=LAB_TEMPLATE)
    assert parent.read_text(encoding='utf-8').startswith(text)
    data = read_toml(file=parent)
    assert data['tool']['uv']['workspace'] == {'members': ['exp']}
    assert data['tool']['uv']['sources'] == {'other': {'workspace': True}}
    assert data['dependency-groups'] == {'dev': ['pytest']}
    assert "add 'exp = { workspace = true }' to [tool.uv.sources]" in caplog.text
    assert 'add "exp" to the dev group' in caplog.text


def test_virtual_workspace_root_gets_only_the_member(tmp_path: Path) -> None:
    parent = write_parent(folder=tmp_path, text='[tool.uv]\npackage = false\n')
    create_project(path=tmp_path / 'exp', template=LAB_TEMPLATE)
    assert read_toml(file=parent) == {
        'tool': {'uv': {'package': False, 'workspace': {'members': ['exp']}}}
    }
    member = read_toml(file=tmp_path / 'exp' / 'pyproject.toml')
    assert member['project']['dependencies'] == ['tslab>=0.1.0a1,<0.2']
    assert 'uv' not in member.get('tool', {})


def test_standalone_layout_leaves_the_parent_alone(tmp_path: Path) -> None:
    parent = write_parent(folder=tmp_path, text='[project]\nname = "my-project"\n')
    before = parent.read_bytes()
    create_project(path=tmp_path / 'exp', template=LAB_TEMPLATE, layout=Layout.STANDALONE)
    assert parent.read_bytes() == before
    assert 'my-project' not in (tmp_path / 'exp' / 'pyproject.toml').read_text(encoding='utf-8')


def test_workspace_layout_without_parent_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(new_module, '_find_parent', lambda *, path: None)  # noqa: ARG005
    assert New(path=tmp_path / 'p', layout=Layout.WORKSPACE).run() == 1
    assert not (tmp_path / 'p').exists()


def test_name_clash_with_parent_fails_before_copying(tmp_path: Path) -> None:
    parent = write_parent(folder=tmp_path, text='[project]\nname = "my_project"\n')
    before = parent.read_bytes()
    assert New(path=tmp_path / 'my-project').run() == 1
    assert not (tmp_path / 'my-project').exists()
    assert parent.read_bytes() == before


def test_unparsable_parent_fails_before_copying(tmp_path: Path) -> None:
    write_parent(folder=tmp_path, text='[project\n')
    assert New(path=tmp_path / 'exp').run() == 1
    assert not (tmp_path / 'exp').exists()


def test_command_line_entry_point_in_a_project(tmp_path: Path) -> None:
    parent = write_parent(folder=tmp_path, text='[project]\nname = "my-project"\n')
    with pytest.raises(SystemExit) as raised:
        main(args=['new', str(tmp_path / 'exp'), '--layout', 'workspace'])
    assert raised.value.code == 0
    assert read_toml(file=parent)['tool']['uv']['workspace'] == {'members': ['exp']}

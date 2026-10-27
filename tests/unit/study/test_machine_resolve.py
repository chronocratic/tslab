"""``resolve_machine``: a ``--machine`` value is a name or a path."""

from pathlib import Path

import pytest

from tslab.study.machine import MachineProfileError, resolve_machine, write_register


@pytest.fixture(autouse=True)
def register(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    file = tmp_path / 'home' / 'machines.toml'
    monkeypatch.setenv('TSLAB_MACHINES_FILE', str(file))
    return file


def _profile(*, folder: Path, name: str) -> Path:
    (folder / 'machines').mkdir(parents=True, exist_ok=True)
    path = folder / 'machines' / f'{name}.toml'
    path.write_text('', encoding='utf-8')
    return path


def test_name_is_found_in_machines_of_the_start_folder(tmp_path: Path) -> None:
    expected = _profile(folder=tmp_path, name='cluster')
    found = resolve_machine(machine='cluster', start=tmp_path)
    assert found.path == expected
    assert found.project_root == tmp_path


def test_nearest_machines_folder_wins_when_projects_nest(tmp_path: Path) -> None:
    _profile(folder=tmp_path, name='cluster')
    inner = _profile(folder=tmp_path / 'experiments', name='cluster')
    found = resolve_machine(machine='cluster', start=tmp_path / 'experiments' / 'src')
    assert found.path == inner
    assert found.project_root == tmp_path / 'experiments'


def test_name_is_found_from_a_nested_folder_above_the_profile(tmp_path: Path) -> None:
    expected = _profile(folder=tmp_path / 'experiments', name='local')
    _profile(folder=tmp_path, name='other')
    deep = tmp_path / 'experiments' / 'a' / 'b'
    deep.mkdir(parents=True)
    assert resolve_machine(machine='local', start=deep).path == expected


def test_registered_name_is_used_when_the_walk_finds_nothing(tmp_path: Path) -> None:
    elsewhere = _profile(folder=tmp_path / 'elsewhere', name='cluster')
    write_register(entries={'cluster': elsewhere})
    start = tmp_path / 'work'
    start.mkdir()
    found = resolve_machine(machine='cluster', start=start)
    assert found.path == elsewhere
    assert found.project_root == start


def test_walk_up_beats_the_register(tmp_path: Path) -> None:
    near = _profile(folder=tmp_path, name='cluster')
    write_register(entries={'cluster': _profile(folder=tmp_path / 'x', name='cluster')})
    assert resolve_machine(machine='cluster', start=tmp_path).path == near


def test_value_with_separator_or_suffix_is_an_absolute_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert resolve_machine(machine='m/local.toml', start=tmp_path).path == (
        tmp_path / 'm' / 'local.toml'
    )
    assert resolve_machine(machine='local.toml', start=tmp_path).path == tmp_path / 'local.toml'


def test_existing_file_without_suffix_is_a_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'profile').write_text('', encoding='utf-8')
    assert resolve_machine(machine='profile', start=tmp_path).path == tmp_path / 'profile'


def test_unknown_name_lists_folders_registered_names_and_the_fix(tmp_path: Path) -> None:
    write_register(entries={'known': tmp_path / 'known.toml'})
    with pytest.raises(MachineProfileError) as error_info:
        resolve_machine(machine='cluster', start=tmp_path)
    text = str(error_info.value)
    assert "'cluster'" in text
    assert str(tmp_path / 'machines') in text
    assert 'known' in text
    assert 'pass the path to the profile file instead' in text
    assert 'tslab machine add <name> <path>' in text


def test_registered_name_with_a_missing_file_has_its_own_error(tmp_path: Path) -> None:
    missing = tmp_path / 'gone' / 'cluster.toml'
    write_register(entries={'cluster': missing})
    with pytest.raises(MachineProfileError) as error_info:
        resolve_machine(machine='cluster', start=tmp_path)
    text = str(error_info.value)
    assert 'cluster' in text
    assert str(missing) in text
    assert 'tslab machine remove cluster' in text

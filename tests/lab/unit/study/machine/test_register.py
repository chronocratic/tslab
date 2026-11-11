"""The machine register: a user-level TOML file mapping names to profile files.

Also how ``--machine`` resolves a name or a path.
"""

import os
from pathlib import Path

import pytest

from tslab.lab.study.machine import (
    MachineProfileError,
    read_register,
    register_file,
    resolve_machine,
    validate_machine_name,
    write_register,
)


@pytest.fixture(autouse=True)
def register(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    file = tmp_path / 'home' / 'machines.toml'
    monkeypatch.setenv('TSLAB_MACHINES_FILE', str(file))
    return file


def test_register_file_follows_the_override_variable(tmp_path: Path) -> None:
    assert register_file(environment={'TSLAB_MACHINES_FILE': str(tmp_path / 'r.toml')}) == (
        tmp_path / 'r.toml'
    )


def test_register_file_uses_xdg_config_home(tmp_path: Path) -> None:
    assert register_file(environment={'XDG_CONFIG_HOME': str(tmp_path)}) == (
        tmp_path / 'tslab' / 'machines.toml'
    )


def test_register_file_from_environment(tmp_path: Path) -> None:
    assert register_file(environment={'XDG_CONFIG_HOME': str(tmp_path)}) == (
        tmp_path / 'tslab' / 'machines.toml'
    )


def test_register_file_falls_back_to_dot_config(tmp_path: Path) -> None:
    assert register_file(environment={'HOME': str(tmp_path)}) == (
        tmp_path / '.config' / 'tslab' / 'machines.toml'
    )


def test_register_file_without_home_is_under_the_expanded_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('HOME', str(tmp_path))
    assert register_file(environment={}) == tmp_path / '.config' / 'tslab' / 'machines.toml'


def test_missing_register_is_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('TSLAB_MACHINES_FILE', str(tmp_path / 'none.toml'))
    assert read_register(environment=os.environ) == {}


def test_register_round_trips_and_creates_the_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('TSLAB_MACHINES_FILE', str(tmp_path / 'new' / 'r.toml'))
    entries = {'b-1': tmp_path / 'b.toml', 'a_2': tmp_path / 'a b.toml'}
    write_register(entries=entries, environment=os.environ)
    assert read_register(environment=os.environ) == entries


def test_malformed_register_fails_loud(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    file = tmp_path / 'r.toml'
    file.write_text('a = 3\n', encoding='utf-8')
    monkeypatch.setenv('TSLAB_MACHINES_FILE', str(file))
    with pytest.raises(MachineProfileError, match=str(file)):
        read_register(environment=os.environ)


@pytest.mark.parametrize('name', ['a/b', 'a.toml', '', 'a b', '..', 'a\\b'])
def test_validate_machine_name_rejects_paths_and_suffixes(name: str) -> None:
    with pytest.raises(MachineProfileError, match='name'):
        validate_machine_name(name=name)


def _profile(*, folder: Path, name: str) -> Path:
    (folder / 'machines').mkdir(parents=True, exist_ok=True)
    path = folder / 'machines' / f'{name}.toml'
    path.write_text('', encoding='utf-8')
    return path


def test_name_is_found_in_machines_of_the_start_folder(tmp_path: Path) -> None:
    expected = _profile(folder=tmp_path, name='cluster')
    found = resolve_machine(machine='cluster', start=tmp_path, environment=os.environ)
    assert found.path == expected
    assert found.project_root == tmp_path


def test_nearest_machines_folder_wins_when_projects_nest(tmp_path: Path) -> None:
    _profile(folder=tmp_path, name='cluster')
    inner = _profile(folder=tmp_path / 'experiments', name='cluster')
    found = resolve_machine(
        machine='cluster', start=tmp_path / 'experiments' / 'src', environment=os.environ
    )
    assert found.path == inner
    assert found.project_root == tmp_path / 'experiments'


def test_name_is_found_from_a_nested_folder_above_the_profile(tmp_path: Path) -> None:
    expected = _profile(folder=tmp_path / 'experiments', name='local')
    _profile(folder=tmp_path, name='other')
    deep = tmp_path / 'experiments' / 'a' / 'b'
    deep.mkdir(parents=True)
    assert resolve_machine(machine='local', start=deep, environment=os.environ).path == expected


def test_registered_name_is_used_when_the_walk_finds_nothing(tmp_path: Path) -> None:
    elsewhere = _profile(folder=tmp_path / 'elsewhere', name='cluster')
    write_register(entries={'cluster': elsewhere}, environment=os.environ)
    start = tmp_path / 'work'
    start.mkdir()
    found = resolve_machine(machine='cluster', start=start, environment=os.environ)
    assert found.path == elsewhere
    assert found.project_root == start


def test_walk_up_beats_the_register(tmp_path: Path) -> None:
    near = _profile(folder=tmp_path, name='cluster')
    write_register(
        entries={'cluster': _profile(folder=tmp_path / 'x', name='cluster')}, environment=os.environ
    )
    assert resolve_machine(machine='cluster', start=tmp_path, environment=os.environ).path == near


def test_value_with_separator_or_suffix_is_an_absolute_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert resolve_machine(machine='m/local.toml', start=tmp_path, environment=os.environ).path == (
        tmp_path / 'm' / 'local.toml'
    )
    assert (
        resolve_machine(machine='local.toml', start=tmp_path, environment=os.environ).path
        == tmp_path / 'local.toml'
    )


def test_existing_file_without_suffix_is_a_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'profile').write_text('', encoding='utf-8')
    assert resolve_machine(machine='profile', start=tmp_path, environment=os.environ).path == (
        tmp_path / 'profile'
    )


def test_unknown_name_lists_the_folder_registered_names_and_the_fix(tmp_path: Path) -> None:
    write_register(entries={'known': tmp_path / 'known.toml'}, environment=os.environ)
    with pytest.raises(MachineProfileError) as error_info:
        resolve_machine(machine='cluster', start=tmp_path, environment=os.environ)
    text = str(error_info.value)
    assert "'cluster'" in text
    assert f'machines/cluster.toml in {tmp_path}' in text
    assert 'registered names are: known' in text
    assert 'Pass a profile path' in text
    assert 'tslab machine add cluster <path>' in text


def test_registered_name_with_a_missing_file_has_its_own_error(tmp_path: Path) -> None:
    missing = tmp_path / 'gone' / 'cluster.toml'
    write_register(entries={'cluster': missing}, environment=os.environ)
    with pytest.raises(MachineProfileError) as error_info:
        resolve_machine(machine='cluster', start=tmp_path, environment=os.environ)
    text = str(error_info.value)
    assert 'cluster' in text
    assert str(missing) in text
    assert 'tslab machine remove cluster' in text

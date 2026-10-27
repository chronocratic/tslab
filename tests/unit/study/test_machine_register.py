"""The machine register: a user-level TOML file mapping names to profile files."""

from pathlib import Path

import pytest

from tslab.study.machine import (
    MachineProfileError,
    read_register,
    register_file,
    validate_machine_name,
    write_register,
)


def test_register_file_follows_the_override_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('TSLAB_MACHINES_FILE', str(tmp_path / 'r.toml'))
    assert register_file() == tmp_path / 'r.toml'


def test_register_file_uses_xdg_config_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv('TSLAB_MACHINES_FILE', raising=False)
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path))
    assert register_file() == tmp_path / 'tslab' / 'machines.toml'


def test_register_file_falls_back_to_dot_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv('TSLAB_MACHINES_FILE', raising=False)
    monkeypatch.delenv('XDG_CONFIG_HOME', raising=False)
    monkeypatch.setenv('HOME', str(tmp_path))
    assert register_file() == tmp_path / '.config' / 'tslab' / 'machines.toml'


def test_missing_register_is_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('TSLAB_MACHINES_FILE', str(tmp_path / 'none.toml'))
    assert read_register() == {}


def test_register_round_trips_and_creates_the_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('TSLAB_MACHINES_FILE', str(tmp_path / 'new' / 'r.toml'))
    entries = {'b-1': tmp_path / 'b.toml', 'a_2': tmp_path / 'a b.toml'}
    write_register(entries=entries)
    assert read_register() == entries


def test_malformed_register_fails_loud(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    file = tmp_path / 'r.toml'
    file.write_text('a = 3\n', encoding='utf-8')
    monkeypatch.setenv('TSLAB_MACHINES_FILE', str(file))
    with pytest.raises(MachineProfileError, match=str(file)):
        read_register()


@pytest.mark.parametrize('name', ['a/b', 'a.toml', '', 'a b', '..', 'a\\b'])
def test_validate_machine_name_rejects_paths_and_suffixes(name: str) -> None:
    with pytest.raises(MachineProfileError, match='name'):
        validate_machine_name(name=name)


def test_validate_machine_name_accepts_a_plain_stem() -> None:
    assert validate_machine_name(name='cluster-2_a') == 'cluster-2_a'

"""``tslab machine add | list | remove``."""

import logging
from pathlib import Path

import pytest

from tslab.cli import main
from tslab.study.machine import read_register

PROFILE = 'launcher = "local"\noutput_root = "runs"\n'


@pytest.fixture(autouse=True)
def register(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    file = tmp_path / 'home' / 'machines.toml'
    monkeypatch.setenv('TSLAB_MACHINES_FILE', str(file))
    return file


def _run(*args: str) -> int:
    with pytest.raises(SystemExit) as exit_info:
        main(args=list(args))
    code = exit_info.value.code
    assert isinstance(code, int)
    return code


def test_add_stores_the_absolute_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'local.toml').write_text(PROFILE, encoding='utf-8')
    assert _run('machine', 'add', 'mine', 'local.toml') == 0
    assert read_register() == {'mine': tmp_path / 'local.toml'}


def test_add_rejects_a_missing_file(tmp_path: Path) -> None:
    assert _run('machine', 'add', 'mine', str(tmp_path / 'none.toml')) == 2
    assert read_register() == {}


def test_add_rejects_a_file_that_is_no_profile(tmp_path: Path) -> None:
    (tmp_path / 'bad.toml').write_text('launcher = "nope"\n', encoding='utf-8')
    assert _run('machine', 'add', 'mine', str(tmp_path / 'bad.toml')) == 2
    assert read_register() == {}


def test_add_rejects_a_bad_name(tmp_path: Path) -> None:
    (tmp_path / 'local.toml').write_text(PROFILE, encoding='utf-8')
    assert _run('machine', 'add', 'a/b', str(tmp_path / 'local.toml')) == 2
    assert _run('machine', 'add', 'a.toml', str(tmp_path / 'local.toml')) == 2


def test_add_replaces_an_existing_name(tmp_path: Path) -> None:
    for name in ('one', 'two'):
        (tmp_path / f'{name}.toml').write_text(PROFILE, encoding='utf-8')
        assert _run('machine', 'add', 'mine', str(tmp_path / f'{name}.toml')) == 0
    assert read_register() == {'mine': tmp_path / 'two.toml'}


def test_list_prints_names_and_marks_missing_files(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / 'local.toml').write_text(PROFILE, encoding='utf-8')
    assert _run('machine', 'add', 'here', str(tmp_path / 'local.toml')) == 0
    (tmp_path / 'gone.toml').write_text(PROFILE, encoding='utf-8')
    assert _run('machine', 'add', 'gone', str(tmp_path / 'gone.toml')) == 0
    (tmp_path / 'gone.toml').unlink()
    with caplog.at_level(logging.INFO):
        assert _run('machine', 'list') == 0
    lines = [record.getMessage() for record in caplog.records]
    assert any('here' in line and str(tmp_path / 'local.toml') in line for line in lines)
    assert any('gone' in line and 'missing' in line for line in lines)


def test_remove_deletes_an_entry_and_unknown_name_fails(tmp_path: Path) -> None:
    (tmp_path / 'local.toml').write_text(PROFILE, encoding='utf-8')
    assert _run('machine', 'add', 'mine', str(tmp_path / 'local.toml')) == 0
    assert _run('machine', 'remove', 'mine') == 0
    assert read_register() == {}
    assert _run('machine', 'remove', 'mine') == 2

"""``tslab machine add | list | remove``."""

import json
import logging
import os
from pathlib import Path

import pytest

from tslab.lab.cli.entrypoint import main
from tslab.lab.study.machine.register import read_register

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
    assert read_register(environment=os.environ) == {'mine': tmp_path / 'local.toml'}


def test_add_rejects_a_missing_file(tmp_path: Path) -> None:
    assert _run('machine', 'add', 'mine', str(tmp_path / 'none.toml')) == 2
    assert read_register(environment=os.environ) == {}


def test_add_rejects_a_file_that_is_no_profile(tmp_path: Path) -> None:
    (tmp_path / 'bad.toml').write_text('launcher = "nope"\n', encoding='utf-8')
    assert _run('machine', 'add', 'mine', str(tmp_path / 'bad.toml')) == 2
    assert read_register(environment=os.environ) == {}


def test_add_rejects_a_bad_name(tmp_path: Path) -> None:
    (tmp_path / 'local.toml').write_text(PROFILE, encoding='utf-8')
    assert _run('machine', 'add', 'a/b', str(tmp_path / 'local.toml')) == 2
    assert _run('machine', 'add', 'a.toml', str(tmp_path / 'local.toml')) == 2


def test_add_replaces_an_existing_name(tmp_path: Path) -> None:
    for name in ('one', 'two'):
        (tmp_path / f'{name}.toml').write_text(PROFILE, encoding='utf-8')
        assert _run('machine', 'add', 'mine', str(tmp_path / f'{name}.toml')) == 0
    assert read_register(environment=os.environ) == {'mine': tmp_path / 'two.toml'}


def test_list_prints_names_and_marks_missing_files(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / 'local.toml').write_text(PROFILE, encoding='utf-8')
    assert _run('machine', 'add', 'here', str(tmp_path / 'local.toml')) == 0
    (tmp_path / 'gone.toml').write_text(PROFILE, encoding='utf-8')
    assert _run('machine', 'add', 'gone', str(tmp_path / 'gone.toml')) == 0
    (tmp_path / 'gone.toml').unlink()
    caplog.clear()
    with caplog.at_level(logging.INFO):
        assert _run('machine', 'list') == 0
    output = caplog.text
    assert 'here' in output
    assert 'local.toml' in output
    assert any('gone' in line and 'missing' in line for line in output.splitlines())


def test_remove_deletes_an_entry_and_unknown_name_fails(tmp_path: Path) -> None:
    (tmp_path / 'local.toml').write_text(PROFILE, encoding='utf-8')
    assert _run('machine', 'add', 'mine', str(tmp_path / 'local.toml')) == 0
    assert _run('machine', 'remove', 'mine') == 0
    assert read_register(environment=os.environ) == {}
    assert _run('machine', 'remove', 'mine') == 2


def test_list_shows_project_profiles_above_the_working_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / 'machines').mkdir()
    (tmp_path / 'machines' / 'local.toml').write_text(PROFILE, encoding='utf-8')
    (tmp_path / 'src').mkdir()
    monkeypatch.chdir(tmp_path / 'src')
    with caplog.at_level(logging.INFO):
        assert _run('machine', 'list') == 0
    lines = caplog.text.splitlines()
    assert any('local' in line and 'project' in line for line in lines)


def test_machine_list_json_lists_project_and_registered(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project_profile = tmp_path / 'machines' / 'project.toml'
    project_profile.parent.mkdir()
    project_profile.write_text(PROFILE, encoding='utf-8')
    registered_profile = tmp_path / 'registered.toml'
    registered_profile.write_text(PROFILE, encoding='utf-8')
    (tmp_path / 'home' / 'machines.toml').parent.mkdir()
    (tmp_path / 'home' / 'machines.toml').write_text(
        f'registered = "{registered_profile.as_posix()}"\n', encoding='utf-8'
    )
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('TSLAB_MACHINES_FILE', str(tmp_path / 'home' / 'machines.toml'))
        assert _run('machine', 'list', '--json') == 0
    document = json.loads(capsys.readouterr().out)
    machines = {entry['name']: entry for entry in document['machines']}
    assert machines['project'] == {
        'name': 'project',
        'source': 'project',
        'profile': project_profile.resolve().as_posix(),
        'note': '',
    }
    assert machines['registered'] == {
        'name': 'registered',
        'source': 'registered',
        'profile': registered_profile.resolve().as_posix(),
        'note': '',
    }


def test_list_without_machines_says_how_to_add_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.chdir(tmp_path)
    with caplog.at_level(logging.INFO):
        assert _run('machine', 'list') == 0
    assert 'tslab machine add' in caplog.text

"""JSON output for commands that support agent use."""

from collections.abc import Sequence
import json
import logging
from pathlib import Path

import pytest

from tslab.lab.cli import main
from tslab.lab.cli.common import print_json


def run_main_raw(*args: str) -> int:
    with pytest.raises(SystemExit) as exit_info:
        main(args=list(args))
    code = exit_info.value.code
    assert isinstance(code, int)
    return code


def test_print_json_writes_one_document(capsys: pytest.CaptureFixture[str]) -> None:
    print_json(document={'schema': 'tslab.x/1', 'a': 1})
    assert json.loads(capsys.readouterr().out) == {'schema': 'tslab.x/1', 'a': 1}


def test_json_mode_sends_logs_to_stderr(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('TSLAB_MACHINES_FILE', str(tmp_path / 'machines.toml'))
    root = logging.getLogger()
    handlers = root.handlers[:]
    level = root.level
    root.handlers.clear()
    try:
        code = run_main_raw('machine', 'list', '--json')
    finally:
        root.handlers[:] = handlers
        root.setLevel(level)
    captured = capsys.readouterr()
    assert code == 0
    assert json.loads(captured.out) == {'schema': 'tslab.machine-list/1', 'machines': []}
    assert 'no machines' in captured.err


def test_json_parse_error_prints_error_document(capsys: pytest.CaptureFixture[str]) -> None:
    code = run_main_raw('machine', 'list', '--json', '--bogus')
    document = json.loads(capsys.readouterr().out)
    assert code == 2
    assert document['schema'] == 'tslab.error/1'
    assert document['exit_code'] == 2


def test_json_usage_error_prints_error_document(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    code = run_main_raw('study', 'status', 'nope:study', '--machine', 'local', '--json')
    assert code == 2
    assert json.loads(capsys.readouterr().out)['exit_code'] == 2


def test_command_name_clash_raises_before_parsing(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(ValueError, match='already lab commands'):
        main(args=['study', '--json'], extra_commands={'study': object})
    assert capsys.readouterr().out == ''


def test_json_command_exception_prints_error_document(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailingCommand:
        def run(self) -> int:
            message = 'unexpected command failure'
            raise RuntimeError(message)

    def parse_command(*, argv: Sequence[str], extra_commands: object) -> FailingCommand:
        del argv, extra_commands
        return FailingCommand()

    monkeypatch.setattr('tslab.lab.cli.entrypoint._parse', parse_command)
    with pytest.raises(RuntimeError, match='unexpected command failure'):
        main(args=['study', 'show', 'toy_studies:study', '--json'])

    document = json.loads(capsys.readouterr().out)
    assert document == {
        'schema': 'tslab.error/1',
        'exit_code': 1,
        'message': 'unexpected command failure',
    }

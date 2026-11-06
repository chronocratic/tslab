"""The internal chain-exit subcommand is callable but absent from group help."""

from importlib import import_module
from pathlib import Path

import pytest

from tests.support.fake_backend import FakeBackend
from tests.unit.cli.study.helpers import toy_project
from tslab.cli import main

_CLI_MODULE = import_module('tslab.cli.study.chain_exit')


def _run_main(*args: str) -> int:
    with pytest.raises(SystemExit) as exit_info:
        main(args=['study', *args])
    code = exit_info.value.code
    assert isinstance(code, int)
    return code


def test_chain_exit_command_parses_internal_arguments(
    tmp_path: Path, fake_backend: FakeBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = toy_project(tmp_path=tmp_path, monkeypatch=monkeypatch)
    assert project.is_dir()
    del fake_backend
    context = object()
    seen: list[tuple[object, int, int]] = []

    def record(*, context: object, job_file: int, index: int) -> int:
        seen.append((context, job_file, index))
        return 0

    monkeypatch.setattr(_CLI_MODULE, '_context_of', lambda **_: context)
    monkeypatch.setattr(_CLI_MODULE, 'chain_exit', record)

    assert (
        _run_main(
            'chain-exit',
            'pkg.studies.chain:study',
            '--machine',
            'cluster.toml',
            '--job-file',
            '3',
            '--index',
            '17',
        )
        == 0
    )
    assert seen == [(context, 3, 17)]


def test_chain_exit_is_hidden_from_study_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(args=['study', '--help'])

    assert exit_info.value.code == 0
    assert 'chain-exit' not in capsys.readouterr().out

import logging
import os
from pathlib import Path
import sys
import time

import pytest

from tslab.jobs.local import start_detached, start_keep_awake, stop_keep_awake


def _write_command(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(f'#!/bin/sh\n{body}\n')
    path.chmod(0o755)
    return path


def _wait_for_file(path: Path) -> None:
    deadline = time.monotonic() + 5
    while not path.exists():
        assert time.monotonic() < deadline, f'the file {path} never appeared'
        time.sleep(0.05)


def _fake_on_path(*, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, body: str) -> None:
    _write_command(tmp_path, name, body)
    monkeypatch.setenv('PATH', tmp_path.as_posix() + os.pathsep + os.environ['PATH'])


def test_keep_awake_passes_pid(*, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args_file = tmp_path / 'args'
    _fake_on_path(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        name='caffeinate',
        body='echo "$@" > "$KA_OUT"\nsleep 30',
    )
    monkeypatch.setattr(sys, 'platform', 'darwin')
    monkeypatch.setenv('KA_OUT', args_file.as_posix())
    process = start_keep_awake(pid=4321, reason='keep the tests running')
    assert process is not None
    _wait_for_file(args_file)
    assert '-i -w 4321' in args_file.read_text()
    stop_keep_awake(process=process)


def test_keep_awake_systemd_inhibit(*, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args_file = tmp_path / 'args'
    _fake_on_path(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        name='systemd-inhibit',
        body='echo "$@" > "$KA_OUT"\nsleep 30',
    )
    monkeypatch.setattr(sys, 'platform', 'linux')
    monkeypatch.setenv('KA_OUT', args_file.as_posix())
    process = start_keep_awake(pid=4321, reason='keep the tests running')
    assert process is not None
    _wait_for_file(args_file)
    assert '--what=idle:sleep --why=keep the tests running tail --pid=4321' in args_file.read_text()
    stop_keep_awake(process=process)


def test_keep_awake_missing_tool_warns(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv('PATH', (tmp_path / 'empty').as_posix())
    with caplog.at_level(logging.WARNING):
        process = start_keep_awake(pid=1, reason='a reason')
    assert process is None
    assert len(caplog.records) == 1
    message = caplog.records[0].message
    assert 'caffeinate' in message or 'systemd-inhibit' in message


def test_keep_awake_failing_helper_warns(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _fake_on_path(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        name='caffeinate',
        body='echo "$@" > "$KA_OUT"\nexit 1',
    )
    monkeypatch.setattr(sys, 'platform', 'darwin')
    monkeypatch.setenv('KA_OUT', (tmp_path / 'args').as_posix())
    process = start_keep_awake(pid=1, reason='a reason')
    assert process is not None
    deadline = time.monotonic() + 5
    while process.poll() is None:
        assert time.monotonic() < deadline, 'the helper did not exit'
        time.sleep(0.05)
    with caplog.at_level(logging.WARNING):
        stop_keep_awake(process=process)
    assert len(caplog.records) == 1
    assert '1' in caplog.records[0].message


def test_start_detached_survives_and_logs(*, tmp_path: Path) -> None:
    log_path = tmp_path / 'log'
    child = start_detached(
        command=[sys.executable, '-c', 'print("ok")'],
        cwd=tmp_path,
        environment=os.environ,
        log_path=log_path,
    )
    assert child.pid is not None
    assert os.getsid(child.pid) == child.pid
    assert child.wait() == 0
    assert log_path.read_text() == 'ok\n'

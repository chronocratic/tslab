import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import pytest

from tslab.jobs.local import boot_id, process_alive, run_process, stop_on_signals


def test_run_process_returns_exit_code_and_lines(*, tmp_path: Path) -> None:
    lines: list[str] = []
    result = run_process(
        command=[sys.executable, '-c', 'print("a"); print("b"); raise SystemExit(3)'],
        cwd=tmp_path,
        environment=os.environ,
        on_line=lines.append,
        on_event=lambda _text: None,
        stop=threading.Event(),
        timeout_seconds=None,
        signal_seconds=0,
    )
    assert lines == ['a\n', 'b\n']
    assert result.exit_code == 3
    assert result.timed_out is False


def test_run_process_sends_usr2_then_term(*, tmp_path: Path) -> None:
    mark = tmp_path / 'mark'
    code = (
        'import os, signal, time\n'
        'def on_usr2(signum, frame):\n'
        '    with open(os.environ["USR2_MARK"], "w") as handle:\n'
        '        handle.write("usr2")\n'
        'signal.signal(signal.SIGUSR2, on_usr2)\n'
        'time.sleep(30)\n'
    )
    events: list[str] = []
    result = run_process(
        command=[sys.executable, '-c', code],
        cwd=tmp_path,
        environment={**os.environ, 'USR2_MARK': mark.as_posix()},
        on_line=lambda _text: None,
        on_event=events.append,
        stop=threading.Event(),
        timeout_seconds=2,
        signal_seconds=1,
    )
    assert mark.read_text() == 'usr2'
    assert result.timed_out is True
    assert len(events) == 2


def test_run_process_kills_after_grace(*, tmp_path: Path) -> None:
    code = 'import signal, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\ntime.sleep(60)\n'
    start = time.monotonic()
    result = run_process(
        command=[sys.executable, '-c', code],
        cwd=tmp_path,
        environment=os.environ,
        on_line=lambda _text: None,
        on_event=lambda _text: None,
        stop=threading.Event(),
        timeout_seconds=1,
        signal_seconds=0,
        grace_seconds=1,
    )
    assert time.monotonic() - start < 4
    assert result.exit_code == -signal.SIGKILL


def test_run_process_stop_event_terminates(*, tmp_path: Path) -> None:
    stop = threading.Event()
    timer = threading.Timer(0.5, stop.set)
    timer.start()
    start = time.monotonic()
    result = run_process(
        command=[sys.executable, '-c', 'import time; time.sleep(60)'],
        cwd=tmp_path,
        environment=os.environ,
        on_line=lambda _text: None,
        on_event=lambda _text: None,
        stop=stop,
        timeout_seconds=None,
        signal_seconds=0,
    )
    assert time.monotonic() - start < 3
    assert result.exit_code == -signal.SIGTERM
    assert result.timed_out is False


def test_run_process_signals_reach_grandchildren(*, tmp_path: Path) -> None:
    pid_file = tmp_path / 'grandchild.pid'
    grandchild = tmp_path / 'grandchild.py'
    grandchild.write_text(
        f'import os, time\nopen({str(pid_file)!r}, "w").write(str(os.getpid()))\ntime.sleep(60)\n'
    )
    parent = tmp_path / 'parent.py'
    parent.write_text(
        f'import subprocess, sys\nsubprocess.call([sys.executable, {str(grandchild)!r}])\n'
    )
    stop = threading.Event()
    timer = threading.Timer(0.5, stop.set)
    timer.start()
    start = time.monotonic()
    result = run_process(
        command=[sys.executable, str(parent)],
        cwd=tmp_path,
        environment=os.environ,
        on_line=lambda _text: None,
        on_event=lambda _text: None,
        stop=stop,
        timeout_seconds=None,
        signal_seconds=0,
    )
    assert time.monotonic() - start < 3
    assert result.exit_code == -signal.SIGTERM
    grandchild_pid = int(pid_file.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(grandchild_pid, 0)


def test_run_process_reports_peak_memory_in_bytes(*, tmp_path: Path) -> None:
    result = run_process(
        command=[sys.executable, '-c', 'import time; x = bytearray(100 * 2 ** 20); time.sleep(1)'],
        cwd=tmp_path,
        environment=os.environ,
        on_line=lambda _text: None,
        on_event=lambda _text: None,
        stop=threading.Event(),
        timeout_seconds=None,
        signal_seconds=0,
    )
    assert result.exit_code == 0
    assert 100 * 2**20 <= result.max_rss_bytes <= 2 * 2**30
    assert result.elapsed_seconds >= 1


def test_stop_on_signals_records_signal() -> None:
    previous = signal.getsignal(signal.SIGUSR1)
    with stop_on_signals(signals=[signal.SIGUSR1]) as request:
        os.kill(os.getpid(), signal.SIGUSR1)
        assert request.event.is_set()
        assert request.signal_number == signal.SIGUSR1
    assert signal.getsignal(signal.SIGUSR1) is previous


def test_boot_id_is_stable_within_a_boot() -> None:
    assert boot_id() == boot_id()  # str on Linux and macOS CI, never raises


def test_process_alive_for_this_process() -> None:
    assert process_alive(pid=os.getpid())


def test_process_alive_for_a_reaped_child() -> None:
    with subprocess.Popen([sys.executable, '-c', 'pass']) as child:
        child.wait()
        pid = child.pid
    assert not process_alive(pid=pid)


def test_run_process_kills_leftovers_that_hold_the_pipe(*, tmp_path: Path) -> None:
    events: list[str] = []
    start = time.monotonic()
    result = run_process(
        command=['sh', '-c', 'sleep 60 & echo started'],
        cwd=tmp_path,
        environment=os.environ,
        on_line=lambda _line: None,
        on_event=events.append,
        stop=threading.Event(),
        timeout_seconds=None,
        signal_seconds=0,
    )
    assert result.exit_code == 0
    assert time.monotonic() - start < 15
    assert any('killed the leftover processes' in event for event in events)


def test_run_process_returns_when_a_leftover_left_the_group(*, tmp_path: Path) -> None:
    script = (
        'import os, subprocess, sys; '
        'subprocess.Popen(["sleep", "30"], start_new_session=True); print("x")'
    )
    start = time.monotonic()
    result = run_process(
        command=[sys.executable, '-c', script],
        cwd=tmp_path,
        environment=os.environ,
        on_line=lambda _line: None,
        on_event=lambda _text: None,
        stop=threading.Event(),
        timeout_seconds=None,
        signal_seconds=0,
    )
    assert result.exit_code == 0
    assert time.monotonic() - start < 20

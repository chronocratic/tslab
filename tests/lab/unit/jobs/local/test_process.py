import contextlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import pytest

from tslab.lab.jobs.local import process_alive, run_process, stop_on_signals
from tslab.lab.jobs.local.process import _stop_group_and_reap


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


def test_run_process_reports_the_pid(*, tmp_path: Path) -> None:
    lines: list[str] = []
    pids: list[int] = []
    run_process(
        command=[sys.executable, '-c', 'import os; print(os.getpid())'],
        cwd=tmp_path,
        environment=os.environ,
        on_line=lines.append,
        on_event=lambda _text: None,
        stop=threading.Event(),
        timeout_seconds=None,
        signal_seconds=0,
        on_spawn=pids.append,
    )
    assert len(pids) == 1
    assert pids[0] == int(lines[0].strip())


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


def _wait_for_file(*, path: Path) -> None:
    deadline = time.monotonic() + 10
    while not path.exists():
        if time.monotonic() > deadline:
            pytest.fail(f'the process did not write {path} in time')
        time.sleep(0.05)


def _wait_until_dead(*, pid: int) -> None:
    # why: a killed process is a zombie until init reaps it, and a zombie still answers kill 0
    deadline = time.monotonic() + 5
    while process_alive(pid=pid):
        if time.monotonic() > deadline:
            pytest.fail(f'the process {pid} did not die in time')
        time.sleep(0.05)


def test_stop_group_and_reap_escalates_to_kill(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ready = tmp_path / 'ready'
    code = (
        'import os, signal, time\n'
        'signal.signal(signal.SIGTERM, signal.SIG_IGN)\n'
        f'open({str(ready)!r}, "w").close()\n'
        'time.sleep(60)\n'
    )
    process = subprocess.Popen(  # noqa: S603 # why: a fixed test command
        [sys.executable, '-c', code], cwd=tmp_path, env=os.environ, start_new_session=True
    )
    try:
        # why: the handler is in before the stop, so the TERM cannot work
        _wait_for_file(path=ready)
        monkeypatch.setattr('tslab.lab.jobs.local.process._LEFTOVER_WAIT_SECONDS', 0.5)
        start = time.monotonic()
        _stop_group_and_reap(pid=process.pid, process=process)
        assert process.returncode == -signal.SIGKILL
        # why: the wait ran, so the TERM was sent and did not stop the child
        assert time.monotonic() - start >= 0.4
    finally:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def test_stop_group_and_reap_kills_descendants_that_survive_term(*, tmp_path: Path) -> None:
    grandchild_pid_file = tmp_path / 'grandchild.pid'
    ready = tmp_path / 'ready'
    grandchild = tmp_path / 'grandchild.py'
    grandchild.write_text(
        'import os, signal, time\n'
        'signal.signal(signal.SIGTERM, signal.SIG_IGN)\n'
        f'open({str(grandchild_pid_file)!r}, "w").write(str(os.getpid()))\n'
        'time.sleep(60)\n'
    )
    leader = tmp_path / 'leader.py'
    leader.write_text(
        'import os, subprocess, sys, time\n'
        f'subprocess.Popen([sys.executable, {str(grandchild)!r}])\n'
        f'while not os.path.exists({str(grandchild_pid_file)!r}):\n'
        '    time.sleep(0.05)\n'
        f'open({str(ready)!r}, "w").close()\n'
        'time.sleep(60)\n'
    )
    process = subprocess.Popen(  # noqa: S603 # why: a fixed test command
        [sys.executable, str(leader)], cwd=tmp_path, env=os.environ, start_new_session=True
    )
    try:
        # why: the grandchild ignores the TERM before the stop, the leader does not
        _wait_for_file(path=ready)
        _stop_group_and_reap(pid=process.pid, process=process)
        grandchild_pid = int(grandchild_pid_file.read_text())
        _wait_until_dead(pid=grandchild_pid)
    finally:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


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


def test_run_process_stops_the_child_when_waiting_is_interrupted(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupted(**_kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr('tslab.lab.jobs.local.process._wait_and_escalate', interrupted)
    pids: list[int] = []
    with pytest.raises(KeyboardInterrupt):
        run_process(
            command=[sys.executable, '-c', 'import time; time.sleep(60)'],
            cwd=tmp_path,
            environment=os.environ,
            on_line=lambda _line: None,
            on_event=lambda _text: None,
            stop=threading.Event(),
            timeout_seconds=None,
            signal_seconds=0,
            on_spawn=pids.append,
        )
    assert not process_alive(pid=pids[0])


@pytest.mark.parametrize('pid', [0, -1])
def test_process_alive_is_false_for_a_non_positive_pid(*, pid: int) -> None:
    assert not process_alive(pid=pid)

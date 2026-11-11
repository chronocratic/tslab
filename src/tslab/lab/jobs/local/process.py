"""Running one process with a timeout escalation and a clean stop."""

from collections.abc import Callable, Generator, Mapping, Sequence
import contextlib
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from types import FrameType
from typing import Any

__all__ = [
    'ProcessResult',
    'StopRequest',
    'boot_id',
    'process_alive',
    'run_process',
    'stop_on_signals',
]

_LOGGER = logging.getLogger(__name__)
_LEFTOVER_WAIT_SECONDS = 5.0


@dataclass(frozen=True, kw_only=True)
class ProcessResult:
    """The outcome of one finished process, with its usage.

    Attributes:
        exit_code: The exit code; the negative signal number when the process
            was killed by a signal.
        elapsed_seconds: Wall time from start to exit.
        cpu_seconds: The user and system CPU time of the process.
        max_rss_bytes: The peak resident memory of the process, in bytes.
        timed_out: Whether the time limit was reached and a TERM was sent.
    """

    exit_code: int
    elapsed_seconds: float
    cpu_seconds: float
    max_rss_bytes: int
    timed_out: bool


class StopRequest:
    """A request to stop a running process, with the signal that asked for it.

    Attributes:
        event: Set as soon as a stop is requested.
        signal_number: The first signal number that requested the stop.
    """

    def __init__(self) -> None:
        """Create a request that no signal has set yet."""
        self.event = threading.Event()
        self.signal_number: int | None = None


@contextlib.contextmanager
def stop_on_signals(*, signals: Sequence[int]) -> Generator[StopRequest]:
    """Catch the given signals for the duration of the block.

    Each caught signal sets the event of the request; the number of the first
    caught signal is recorded. The handlers installed on entry are restored
    on exit, also when the block raises.

    Args:
        signals: The signal numbers to catch.

    Yields:
        The request the handlers set.

    Raises:
        RuntimeError: If called from a thread other than the main thread:
            signal handlers can only be installed from the main thread.
    """
    if threading.current_thread() is not threading.main_thread():
        message = (
            'stop_on_signals only works from the main thread: '
            'signal handlers can only be installed from the main thread'
        )
        raise RuntimeError(message)
    request = StopRequest()

    # why: the interpreter calls signal handlers with positional arguments
    def handle(signum: int, _frame: FrameType | None) -> None:
        if request.signal_number is None:
            request.signal_number = signum
        request.event.set()

    # why: a previous handler is a callable, the SIG_DFL or SIG_IGN sentinel (an int), or None
    old_handlers: dict[int, Callable[[int, FrameType | None], None] | int | None] = {}
    try:
        for signum in signals:
            old_handlers[signum] = signal.signal(signum, handle)
        yield request
    finally:
        for signum, previous in old_handlers.items():
            signal.signal(signum, previous)


def _at_timeout(*, elapsed: float, timeout_seconds: float | None) -> bool:
    """Whether the time limit is set and reached at ``elapsed``."""
    return timeout_seconds is not None and elapsed >= timeout_seconds


def _usr2_due(*, elapsed: float, timeout_seconds: float | None, signal_seconds: float) -> bool:
    """Whether the time limit asks for a USR2 at ``elapsed``."""
    return (
        signal_seconds > 0
        and timeout_seconds is not None
        and elapsed >= timeout_seconds - signal_seconds
    )


def _term_due(*, elapsed: float, stop: threading.Event, timeout_seconds: float | None) -> bool:
    """Whether the stop event or the time limit asks for a TERM at ``elapsed``."""
    return stop.is_set() or _at_timeout(elapsed=elapsed, timeout_seconds=timeout_seconds)


def _kill_due(*, now: float, term_at: float | None, grace_seconds: float) -> bool:
    """Whether the grace period after a TERM is over."""
    return term_at is not None and now - term_at >= grace_seconds


def _wait_and_escalate(  # noqa: PLR0913 # why: one keyword per escalation setting
    *,
    pid: int,
    stop: threading.Event,
    timeout_seconds: float | None,
    signal_seconds: float,
    grace_seconds: float,
    on_event: Callable[[str], None],
) -> ProcessResult:
    """Wait for one process to exit, escalating signals at the time limit.

    Args:
        pid: The pid of the process; the leader of its session.
        stop: An event that requests a stop at any time.
        timeout_seconds: The time limit, or ``None`` for no limit.
        signal_seconds: Seconds before the limit to send ``SIGUSR2``;
            zero disables the signal.
        grace_seconds: Seconds to wait after a ``SIGTERM`` before a ``SIGKILL``.
        on_event: Called when a signal is sent to the process group.

    Returns:
        The exit code and the usage of the finished process.
    """

    def send(*, signum: int, name: str) -> None:
        try:
            os.killpg(pid, signum)
        except ProcessLookupError:
            # why: the group already exited between the wait4 poll and the signal
            return
        on_event(f'sent {name} to the process group of {pid}')

    start = time.monotonic()
    sent_usr2 = False
    sent_term = False
    sent_kill = False
    term_at: float | None = None
    timed_out = False
    while True:
        # why: wait4 reaps the child, so Popen.wait and Popen.poll would eat
        # the usage and must not be called
        result = os.wait4(pid, os.WNOHANG)
        if result[0] != 0:
            break
        now = time.monotonic()
        elapsed = now - start
        if not sent_usr2 and _usr2_due(
            elapsed=elapsed, timeout_seconds=timeout_seconds, signal_seconds=signal_seconds
        ):
            sent_usr2 = True
            send(signum=signal.SIGUSR2, name='USR2')
        if not sent_term and _term_due(elapsed=elapsed, stop=stop, timeout_seconds=timeout_seconds):
            sent_term = True
            term_at = now
            timed_out = _at_timeout(elapsed=elapsed, timeout_seconds=timeout_seconds)
            send(signum=signal.SIGTERM, name='TERM')
        if not sent_kill and _kill_due(now=now, term_at=term_at, grace_seconds=grace_seconds):
            sent_kill = True
            send(signum=signal.SIGKILL, name='KILL')
        time.sleep(0.2)
    _, state, usage = result
    max_rss_bytes = usage.ru_maxrss * _RSS_BYTES_PER_UNIT.get(sys.platform, 1024)
    return ProcessResult(
        exit_code=os.waitstatus_to_exitcode(state),
        elapsed_seconds=time.monotonic() - start,
        cpu_seconds=usage.ru_utime + usage.ru_stime,
        max_rss_bytes=max_rss_bytes,
        timed_out=timed_out,
    )


# why: ru_maxrss is in bytes on macOS and in kilobytes elsewhere
_RSS_BYTES_PER_UNIT: Mapping[str, int] = {'darwin': 1}


def _stop_group_and_reap(*, pid: int, process: subprocess.Popen[Any]) -> None:
    """Stop the process group of ``pid`` and reap its leader.

    The group gets a ``SIGTERM``; once the leader has not exited within
    ``_LEFTOVER_WAIT_SECONDS``, the group gets a ``SIGKILL``, so a leader
    that survives the ``SIGTERM`` does not keep the cleanup waiting and its
    descendants do not outlive it.

    Args:
        pid: The pid of the process; the leader of its group.
        process: The child to reap; its pipes are not read, so the pipe
            type does not matter.
    """
    with contextlib.suppress(ProcessLookupError):
        # why: the group, not the child, so grandchildren do not survive either
        os.killpg(pid, signal.SIGTERM)
    deadline = time.monotonic() + _LEFTOVER_WAIT_SECONDS
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.1)
    with contextlib.suppress(ProcessLookupError, PermissionError):
        # why: the leader may obey the TERM and be gone; the KILL still reaches
        # the descendants that survive it (macOS answers EPERM, not ESRCH, for
        # a group left with only a zombie)
        os.killpg(pid, signal.SIGKILL)
    process.wait()


def _report_spawn(
    *, on_spawn: Callable[[int], None] | None, pid: int, process: subprocess.Popen[str]
) -> None:
    """Call ``on_spawn`` with the pid of the child that started.

    When the callback raises, the process group of the child is stopped and
    reaped before the error propagates, so a failure of the callback never
    leaves a live process without a watcher.

    Args:
        on_spawn: The callback, or ``None`` for none.
        pid: The pid of the child.
        process: The child to stop and reap when the callback raises.

    Raises:
        Exception: Whatever the callback raised, once the child is gone.
    """
    if on_spawn is None:
        return
    try:
        on_spawn(pid)
    except Exception:  # why: a callback failure must never leave the child running
        _stop_group_and_reap(pid=pid, process=process)
        raise


def run_process(  # noqa: PLR0913 # why: keyword-only public API; each option is documented below
    *,
    command: Sequence[str],
    cwd: Path,
    environment: Mapping[str, str],
    on_line: Callable[[str], None],
    on_event: Callable[[str], None],
    stop: threading.Event,
    timeout_seconds: float | None,
    signal_seconds: float,
    grace_seconds: float = 30.0,
    pass_fds: tuple[int, ...] = (),
    on_spawn: Callable[[int], None] | None = None,
) -> ProcessResult:
    """Run one process in its own session, escalating signals at the time limit.

    The process is the leader of a new session, so every signal goes to the
    whole process group, grandchildren included.

    Args:
        command: The command and its arguments.
        cwd: The working directory of the process.
        environment: The full environment of the process.
        on_line: Called for each line the process writes to stdout or stderr.
        on_event: Called when a signal is sent to the process group.
        stop: An event that requests a stop at any time.
        timeout_seconds: The time limit, or ``None`` for no limit.
        signal_seconds: Seconds before the limit to send ``SIGUSR2``;
            zero disables the signal.
        grace_seconds: Seconds to wait after a ``SIGTERM`` before a ``SIGKILL``.
        pass_fds: Extra file descriptors to keep open in the process.
        on_spawn: Called with the pid of the child right after it starts,
            or ``None`` for no callback.

    Returns:
        The exit code and the usage of the finished process.

    Raises:
        OSError: If the command cannot be started.
    """
    process = subprocess.Popen(  # noqa: S603 # why: the command comes from the caller; running it is the purpose
        command,
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding='utf-8',
        errors='replace',
        start_new_session=True,
        pass_fds=pass_fds,
    )
    pid = process.pid
    if pid is None:
        # why: Popen sets the pid before it returns; the check keeps the type narrow
        message = 'the child process has no pid'
        raise RuntimeError(message)
    _report_spawn(on_spawn=on_spawn, pid=pid, process=process)

    def read_lines() -> None:
        stdout = process.stdout
        if stdout is None:
            # why: Popen is created with a pipe; the check keeps the type narrow
            message = 'the child process stdout is missing'
            raise RuntimeError(message)
        failed = False
        for line in stdout:
            if failed:
                continue
            try:
                on_line(line)
            except Exception:
                # why: a dead reader stops draining the pipe and the child blocks on a full
                # buffer; keep reading and drop the lines until the pipe closes
                failed = True
                _LOGGER.exception('could not record the output of the child; dropping it')

    reader = threading.Thread(target=read_lines, name='process-stdout', daemon=True)
    reader.start()
    try:
        result = _wait_and_escalate(
            pid=pid,
            stop=stop,
            timeout_seconds=timeout_seconds,
            signal_seconds=signal_seconds,
            grace_seconds=grace_seconds,
            on_event=on_event,
        )
    except BaseException:  # why: KeyboardInterrupt too; no child may outlive the lease
        _stop_group_and_reap(pid=pid, process=process)
        raise
    reader.join(timeout=_LEFTOVER_WAIT_SECONDS)
    if reader.is_alive():
        # why: the process group may exit before the kill reaches it
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pid, signal.SIGKILL)
        on_event(f'killed the leftover processes of {pid}')
        reader.join(timeout=_LEFTOVER_WAIT_SECONDS)
    if not reader.is_alive() and process.stdout is not None:
        process.stdout.close()
    return result


def _boot_id_from_proc() -> str | None:
    """The boot id of a Linux host, read from ``/proc``."""
    try:
        return Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    except OSError:
        return None


def _boot_id_from_sysctl() -> str | None:
    """The boot time of a macOS host, read with ``sysctl``."""
    try:
        result = subprocess.run(
            args=('sysctl', '-n', 'kern.boottime'),  # noqa: S607 # why: a fixed system command from the path
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


_BOOT_ID_BY_PLATFORM: Mapping[str, Callable[[], str | None]] = {
    'linux': _boot_id_from_proc,
    'darwin': _boot_id_from_sysctl,
}


def boot_id() -> str | None:
    """The identity of the current boot, or ``None`` when it cannot be read.

    On Linux the value comes from ``/proc/sys/kernel/random/boot_id``; on
    macOS from ``sysctl -n kern.boottime``, waiting at most 5 seconds. Any
    other platform, a read error, a timeout or a non-zero exit gives
    ``None``, so a check that compares boots falls back to the pid alone.

    Returns:
        The boot identity, or ``None`` when it cannot be read.
    """
    reader = _BOOT_ID_BY_PLATFORM.get(sys.platform)
    if reader is None:
        return None
    return reader()


def process_alive(*, pid: int) -> bool:
    """Return whether a process with ``pid`` still exists on this host.

    A ``PermissionError`` means the process exists but belongs to another
    user, so it counts as alive.

    Args:
        pid: The process id to check; zero or a negative id is never a process.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True

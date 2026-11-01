"""Running one process with a timeout escalation and a clean stop."""

from collections.abc import Callable, Generator, Mapping, Sequence
import contextlib
from dataclasses import dataclass
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from types import FrameType

__all__ = ['ProcessResult', 'StopRequest', 'process_alive', 'run_process', 'stop_on_signals']


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
    if sys.platform == 'darwin':
        max_rss_bytes = usage.ru_maxrss
    else:
        max_rss_bytes = usage.ru_maxrss * 1024
    return ProcessResult(
        exit_code=os.waitstatus_to_exitcode(state),
        elapsed_seconds=time.monotonic() - start,
        cpu_seconds=usage.ru_utime + usage.ru_stime,
        max_rss_bytes=max_rss_bytes,
        timed_out=timed_out,
    )


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

    def read_lines() -> None:
        stdout = process.stdout
        if stdout is None:
            # why: Popen is created with a pipe; the check keeps the type narrow
            message = 'the child process stdout is missing'
            raise RuntimeError(message)
        for line in stdout:
            on_line(line)

    reader = threading.Thread(target=read_lines, name='process-stdout', daemon=True)
    reader.start()
    result = _wait_and_escalate(
        pid=pid,
        stop=stop,
        timeout_seconds=timeout_seconds,
        signal_seconds=signal_seconds,
        grace_seconds=grace_seconds,
        on_event=on_event,
    )
    reader.join()
    return result


def process_alive(*, pid: int) -> bool:
    """Return whether a process with ``pid`` still exists on this host.

    A ``PermissionError`` means the process exists but belongs to another
    user, so it counts as alive.

    Args:
        pid: The process id to check.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True

"""A watchdog that kills the process groups it still holds when its stdin closes."""

from collections.abc import Iterable, Sequence
import contextlib
from dataclasses import dataclass
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Self

__all__ = ['Watchdog', 'main']


@dataclass(frozen=True, kw_only=True)
class Watchdog:
    """A detached process that kills the process groups its owner still runs.

    The watchdog reads lines from its stdin: ``+pid`` adds the process group
    of ``pid`` to the ones it holds, ``-pid`` drops it. When the pipe closes,
    the owner is gone; the watchdog sends SIGTERM to every group it still
    holds, waits ``grace_seconds``, then sends SIGKILL to the groups that are
    still alive.

    Attributes:
        process: The watchdog process; the caller writes the lines to its stdin.
        grace_seconds: Seconds between the SIGTERM and the SIGKILL.
    """

    process: subprocess.Popen[bytes]
    grace_seconds: float

    @classmethod
    def start(cls, *, grace_seconds: float) -> Self:
        """Start a watchdog in its own session.

        The script runs by path, not as a module, so importing the package is
        not needed.

        Args:
            grace_seconds: Seconds between the SIGTERM and the SIGKILL.

        Returns:
            The started watchdog.
        """
        script = Path(__file__).resolve()
        process = subprocess.Popen(  # noqa: S603 # why: the command is the interpreter and the script of this file
            [sys.executable, script.as_posix(), str(grace_seconds)],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return cls(process=process, grace_seconds=grace_seconds)

    def watch(self, *, pid: int) -> None:
        """Tell the watchdog to hold the process group of ``pid``.

        Args:
            pid: The pid of the group leader to hold.
        """
        self._send(line=f'+{pid}\n')

    def forget(self, *, pid: int) -> None:
        """Tell the watchdog to drop the process group of ``pid``.

        Args:
            pid: The pid of the group leader to drop.
        """
        self._send(line=f'-{pid}\n')

    def _send(self, *, line: str) -> None:
        stdin = self.process.stdin
        if stdin is None:
            # why: the process is started with a pipe; the check keeps the type narrow
            message = 'the watchdog stdin is missing'
            raise RuntimeError(message)
        with contextlib.suppress(BrokenPipeError):
            stdin.write(line.encode('utf-8'))
            stdin.flush()

    def close(self) -> None:
        """Close the pipe and wait for the watchdog to exit.

        The watchdog gets at most ``grace_seconds`` plus 10 seconds to kill
        the groups it holds; after that it is killed itself.
        """
        stdin = self.process.stdin
        if stdin is not None:
            stdin.close()
        deadline = time.monotonic() + self.grace_seconds + 10
        while self.process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.2)
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()


def main(*, argv: Sequence[str], lines: Iterable[str]) -> int:
    """Hold the watched process groups, then kill them when the input ends.

    Args:
        argv: The arguments of the script; the first is the grace in seconds.
        lines: The lines of stdin, read one at a time.

    Returns:
        0 once every held group is gone.
    """
    grace_seconds = float(argv[0])
    watched: set[int] = set()
    for line in lines:
        watched = _apply_line(line=line, watched=watched)
    _stop_groups(pids=tuple(watched), grace_seconds=grace_seconds)
    return 0


def _apply_line(*, line: str, watched: set[int]) -> set[int]:
    """The watched pids after one ``+pid`` or ``-pid`` line."""
    text = line.strip()
    if not text[1:].isdigit() or text[0] not in {'+', '-'}:
        return watched
    pid = int(text[1:])
    if text[0] == '+':
        updated = set(watched)
        updated.add(pid)
        return updated
    return watched - {pid}


def _stop_groups(*, pids: tuple[int, ...], grace_seconds: float) -> None:
    """Send SIGTERM to every group, then SIGKILL to the ones still alive."""
    for pid in pids:
        _signal_group(pid=pid, signum=signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    alive = pids
    while time.monotonic() < deadline:
        alive = tuple(pid for pid in alive if _group_exists(pid=pid))
        if not alive:
            break
        time.sleep(0.2)
    for pid in alive:
        _signal_group(pid=pid, signum=signal.SIGKILL)


def _group_exists(*, pid: int) -> bool:
    """Whether the process group of ``pid`` still exists and is ours."""
    try:
        os.killpg(pid, 0)
    except (ProcessLookupError, PermissionError):
        # why: a group of another user is not ours and must be left alone
        return False
    return True


def _signal_group(*, pid: int, signum: int) -> None:
    """Send one signal to the process group of ``pid``, ignoring a gone group."""
    try:
        os.killpg(pid, signum)
    except (ProcessLookupError, PermissionError):
        # why: the group may exit or leave between the check and the signal
        return


if __name__ == '__main__':
    sys.exit(main(argv=sys.argv[1:], lines=sys.stdin))

"""Keep the machine awake and start processes in a new session."""

from collections.abc import Mapping, Sequence
import logging
from pathlib import Path
import shutil
import subprocess
import sys

__all__ = ['start_detached', 'start_keep_awake', 'stop_keep_awake']

_LOGGER = logging.getLogger(__name__)


def _keep_awake_command(*, pid: int, reason: str) -> list[str] | None:
    """The keep-awake command of this platform, or ``None`` when no tool is found.

    Args:
        pid: The pid of the process to keep awake for.
        reason: A short reason string shown by ``systemd-inhibit``.

    Returns:
        The tool and its arguments, or ``None`` after a warning.
    """
    if sys.platform == 'darwin':
        tool = shutil.which('caffeinate')
        arguments = ['-i', '-w', str(pid)]
    else:
        tool = shutil.which('systemd-inhibit')
        arguments = [
            '--what=idle:sleep',
            f'--why={reason}',
            'tail',
            f'--pid={pid}',
            '-f',
            '/dev/null',
        ]
    if tool is None:
        name = 'caffeinate' if sys.platform == 'darwin' else 'systemd-inhibit'
        _LOGGER.warning('no %s found; the machine may sleep while the process runs', name)
        return None
    return [tool, *arguments]


def start_keep_awake(*, pid: int, reason: str) -> subprocess.Popen[bytes] | None:
    """Start a helper that keeps the machine awake while ``pid`` runs.

    On Darwin the helper is ``caffeinate -i -w <pid>``. Elsewhere it is
    ``systemd-inhibit`` watching the pid with ``tail``, when the tool is
    found. The output of the helper goes to ``DEVNULL``.

    Args:
        pid: The pid of the process to keep awake for.
        reason: A short reason string shown by ``systemd-inhibit``.

    Returns:
        The helper process, or ``None`` when no keep-awake tool is found.
    """
    command = _keep_awake_command(pid=pid, reason=reason)
    if command is None:
        return None
    return subprocess.Popen(  # noqa: S603 # why: the command is a fixed platform tool, not user input
        command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def stop_keep_awake(*, process: subprocess.Popen[bytes] | None) -> None:
    """Stop the keep-awake helper, with one warning when it already failed.

    Args:
        process: The helper started by ``start_keep_awake``, or ``None``.

    Returns:
        None.
    """
    if process is None:
        return
    code = process.poll()
    if code is None:
        process.terminate()
        process.wait()
        return
    if code != 0:
        _LOGGER.warning('the keep-awake helper exited with code %s', code)


def start_detached(
    *, command: Sequence[str], cwd: Path, environment: Mapping[str, str], log_path: Path
) -> subprocess.Popen[bytes]:
    """Start a process in a new session that appends its output to a log file.

    The process is the leader of a new session and reads nothing from stdin,
    so it keeps running when the caller exits.

    Args:
        command: The command and its arguments.
        cwd: The working directory of the process.
        environment: The full environment of the process.
        log_path: The file that stdout and stderr are appended to.

    Returns:
        The started process.

    Raises:
        OSError: If the command cannot be started.
    """
    with log_path.open('ab') as log_file:
        return subprocess.Popen(  # noqa: S603 # why: the command comes from the caller; running it is the purpose
            command,
            cwd=cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

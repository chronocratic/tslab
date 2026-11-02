"""The pid file of a local launch: which tasks of the study run, wait or ended."""

from collections.abc import Generator, Sequence
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
from typing import cast

from tslab.core.files import atomic_write_json, read_json_object
from tslab.core.meta import hostname
from tslab.jobs.local import process_alive, start_detached, tail_lines
from tslab.study.launchers import OnTaskState

__all__ = [
    'LAUNCH_FILE',
    'LAUNCH_LOG',
    'LocalLaunch',
    'LocalLaunchError',
    'detach_launch',
    'hold_launch',
    'launch_is_live',
    'local_job_states',
    'read_launch',
]

LAUNCH_FILE = 'local-launch.json'
LAUNCH_LOGS = 'local_logs'
DETACHED_VARIABLE = 'TSLAB_DETACHED'
_JOB_STATES = frozenset(('pending', 'running', 'ended'))
_LOGGER = logging.getLogger(__name__)


class LocalLaunchError(RuntimeError):
    """A local launch cannot start or be read because of the launch file."""


@dataclass(frozen=True, kw_only=True)
class LocalLaunch:
    """The launch recorded in the pid file.

    Args:
        pid: The process that runs the launch.
        hostname: The host of that process.
        started: The start time, UTC ISO.
        tasks: Task id and state (``pending``, ``running`` or ``ended``) pairs.
    """

    pid: int
    hostname: str
    started: str
    tasks: tuple[tuple[str, str], ...]


def read_launch(*, study_folder: Path) -> LocalLaunch | None:
    """The launch in the study folder, or ``None`` when there is no file.

    Args:
        study_folder: The ``.study`` folder of the study.

    Raises:
        LocalLaunchError: If the file is not a launch file.
    """
    path = study_folder / LAUNCH_FILE
    try:
        document = read_json_object(path=path)
    except FileNotFoundError:
        return None
    try:
        return LocalLaunch(
            pid=int(str(document['pid'])),
            hostname=str(document['hostname']),
            started=str(document['started']),
            tasks=tuple(
                (str(key), str(state)) for key, state in cast('list[list[str]]', document['tasks'])
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        message = f'{path} is not a local launch file ({error}); remove it'
        raise LocalLaunchError(message) from error


def launch_is_live(*, launch: LocalLaunch, host: str) -> bool:
    """Whether the process of the launch exists on this host.

    Args:
        launch: The launch to check.
        host: The current host name.
    """
    if launch.hostname != host:
        return False
    return process_alive(pid=launch.pid)


def _write(*, path: Path, launch: LocalLaunch) -> None:
    atomic_write_json(
        path=path,
        document={
            'pid': launch.pid,
            'hostname': launch.hostname,
            'started': launch.started,
            'tasks': [list(item) for item in launch.tasks],
        },
    )


def _check_free(*, study_folder: Path, host: str) -> None:
    previous = read_launch(study_folder=study_folder)
    if previous is None:
        return
    path = study_folder / LAUNCH_FILE
    if previous.hostname != host:
        message = (
            f'a launch on host {previous.hostname} holds {path}; remove the file if it is over'
        )
        raise LocalLaunchError(message)
    if launch_is_live(launch=previous, host=host):
        message = (
            f'a local launch is running (pid {previous.pid}); '
            'wait for it or stop it with tslab study cancel'
        )
        raise LocalLaunchError(message)
    _LOGGER.warning('removing the stale launch file of pid %s', previous.pid)
    path.unlink()


@contextlib.contextmanager
def hold_launch(
    *, study_folder: Path, task_ids: Sequence[str], pid: int, host: str, now: str
) -> Generator[OnTaskState]:
    """Hold the pid file of the study while a launch runs.

    Args:
        study_folder: The ``.study`` folder of the study.
        task_ids: The ids of the tasks the launch will run; all start ``pending``.
        pid: The process id of the launch.
        host: The host name of the launch.
        now: The UTC ISO start time of the launch.

    Yields:
        The setter, called with ``task_id=`` and ``state=``, that rewrites
        the file; safe across threads.

    Raises:
        LocalLaunchError: If a live launch, or a launch of another host, holds the file.
    """
    _check_free(study_folder=study_folder, host=host)
    path = study_folder / LAUNCH_FILE
    states = dict.fromkeys(task_ids, 'pending')
    started = now
    lock = threading.Lock()

    def write() -> None:
        _write(
            path=path,
            launch=LocalLaunch(
                pid=pid, hostname=host, started=started, tasks=tuple(states.items())
            ),
        )

    def set_state(*, task_id: str, state: str) -> None:
        with lock:
            states[task_id] = state
            write()

    study_folder.mkdir(parents=True, exist_ok=True)
    write()
    try:
        yield set_state
    finally:
        path.unlink(missing_ok=True)


def local_job_states(*, study_folder: Path, host: str) -> dict[str, str]:
    """The job state of every task of a live launch; empty without one.

    Args:
        study_folder: The ``.study`` folder of the study.
        host: The current host name.
    """
    launch = read_launch(study_folder=study_folder)
    if launch is None or not launch_is_live(launch=launch, host=host):
        return {}
    return {key: state for key, state in launch.tasks if state in _JOB_STATES}


def _log_tail(*, path: Path, lines: int = 20) -> str:
    try:
        tail = tail_lines(path=path, lines=lines)
    except OSError:
        # why: tail_lines only returns None for a missing file; other read errors
        # (for example permissions) must not stop the launch report
        return ''
    return '\n'.join(tail) if tail is not None else ''


def _stop_detached(*, process: subprocess.Popen[bytes]) -> None:
    """Stop the detached launch process group, then wait for its leader."""
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    process.wait()


def detach_launch(  # why: keyword-only public API; each option is documented below
    *,
    study_folder: Path,
    arguments: Sequence[str],
    project_root: Path,
    wait_command: str,
    timeout_seconds: float = 30.0,
) -> int:
    """Start ``tslab`` with the arguments in a new session and wait until it holds the pid file.

    Args:
        study_folder: The ``.study`` folder of the study.
        arguments: The ``tslab`` arguments of the launch, without ``--detach``.
        project_root: The folder the process runs in.
        wait_command: The command shown to follow the launch.
        timeout_seconds: How long to wait for the pid file; 30 seconds by default.

    Returns:
        0 once the process holds the pid file; its exit code if it ended first;
        1 on a launch file problem, a start failure or a timeout.
    """
    try:
        _check_free(study_folder=study_folder, host=hostname())
        study_folder.mkdir(parents=True, exist_ok=True)
        log_path = study_folder / LAUNCH_LOG
        process = start_detached(
            command=[sys.executable, '-m', 'tslab', *arguments],
            cwd=project_root,
            environment=os.environ,
            log_path=log_path,
        )
    except (LocalLaunchError, OSError) as error:
        _LOGGER.error('%s', error)  # noqa: TRY400 # why: a launch problem is shown, not raised
        return 1
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        launch = read_launch(study_folder=study_folder)
        if launch is not None and launch.pid == process.pid:
            _LOGGER.info('local launch %s running; log: %s', process.pid, log_path)
            _LOGGER.info('next: %s', wait_command)
            return 0
        code = process.poll()
        if code is not None:
            _LOGGER.error('the launch ended with exit code %s:\n%s', code, _log_tail(path=log_path))
            return code
        time.sleep(0.1)
    _stop_detached(process=process)
    _LOGGER.error(
        'the launch did not start in %s seconds and was stopped:\n%s',
        timeout_seconds,
        _log_tail(path=log_path),
    )
    return 1

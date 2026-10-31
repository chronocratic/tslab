"""The pid file of a local launch: which tasks of the study run, wait or ended."""

from collections.abc import Callable, Generator, Sequence
import contextlib
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import sys
import threading
import time
from typing import cast

from tslab.core.meta import hostname
from tslab.core.paths import atomic_write_json, read_json_object
from tslab.core.status import utc_now_iso
from tslab.jobs.local import start_detached
from tslab.study.state import LOCAL_ENDED

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
LAUNCH_LOG = 'local-launch.log'
_JOB_STATES = {'pending': 'PENDING', 'running': 'RUNNING', 'ended': LOCAL_ENDED}
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


def launch_is_live(*, launch: LocalLaunch) -> bool:
    """Whether the process of the launch exists on this host.

    Args:
        launch: The launch to check.
    """
    if launch.hostname != hostname():
        return False
    try:
        os.kill(launch.pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


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


def _check_free(*, study_folder: Path) -> None:
    previous = read_launch(study_folder=study_folder)
    if previous is None:
        return
    path = study_folder / LAUNCH_FILE
    if previous.hostname != hostname():
        message = (
            f'a launch on host {previous.hostname} holds {path}; remove the file if it is over'
        )
        raise LocalLaunchError(message)
    if launch_is_live(launch=previous):
        message = (
            f'a local launch is running (pid {previous.pid}); '
            'wait for it or stop it with tslab study cancel'
        )
        raise LocalLaunchError(message)
    _LOGGER.warning('removing the stale launch file of pid %s', previous.pid)
    path.unlink()


@contextlib.contextmanager
def hold_launch(
    *, study_folder: Path, task_ids: Sequence[str]
) -> Generator[Callable[[str, str], None]]:
    """Hold the pid file of the study while a launch runs.

    Args:
        study_folder: The ``.study`` folder of the study.
        task_ids: The ids of the tasks the launch will run; all start ``pending``.

    Yields:
        A setter ``(task_id, state)`` that rewrites the file; safe across threads.

    Raises:
        LocalLaunchError: If a live launch, or a launch of another host, holds the file.
    """
    _check_free(study_folder=study_folder)
    path = study_folder / LAUNCH_FILE
    states = dict.fromkeys(task_ids, 'pending')
    started = utc_now_iso()
    lock = threading.Lock()

    def write() -> None:
        _write(
            path=path,
            launch=LocalLaunch(
                pid=os.getpid(), hostname=hostname(), started=started, tasks=tuple(states.items())
            ),
        )

    def set_state(task_id: str, state: str) -> None:
        with lock:
            states[task_id] = state
            write()

    study_folder.mkdir(parents=True, exist_ok=True)
    write()
    try:
        yield set_state
    finally:
        path.unlink(missing_ok=True)


def local_job_states(*, study_folder: Path) -> dict[str, str]:
    """The job state of every task of a live launch; empty without one.

    Args:
        study_folder: The ``.study`` folder of the study.
    """
    launch = read_launch(study_folder=study_folder)
    if launch is None or not launch_is_live(launch=launch):
        return {}
    return {key: _JOB_STATES[state] for key, state in launch.tasks if state in _JOB_STATES}


def _log_tail(*, path: Path, lines: int = 20) -> str:
    try:
        return '\n'.join(path.read_text(encoding='utf-8', errors='replace').splitlines()[-lines:])
    except OSError:
        return ''


def detach_launch(  # why: keyword-only public API; each option is documented below
    *,
    study_folder: Path,
    arguments: Sequence[str],
    project_root: Path,
    wait_command: str,
    timeout_seconds: float = 10.0,
) -> int:
    """Start ``tslab`` with the arguments in a new session and wait until it holds the pid file.

    Args:
        study_folder: The ``.study`` folder of the study.
        arguments: The ``tslab`` arguments of the launch, without ``--detach``.
        project_root: The folder the process runs in.
        wait_command: The command shown to follow the launch.
        timeout_seconds: How long to wait for the pid file.

    Returns:
        0 once the process holds the pid file; its exit code if it ended first;
        1 on a launch file problem, a start failure or a timeout.
    """
    try:
        _check_free(study_folder=study_folder)
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
    _LOGGER.error(
        'the launch did not start in %s seconds:\n%s', timeout_seconds, _log_tail(path=log_path)
    )
    return 1

"""The lock of a run folder: one live process per folder, with a heartbeat.

Entering a run claims its folder with a ``.lock`` file created atomically. The
owner refreshes the file from a heartbeat thread, so any host can tell from
the modification time whether the owner is still alive. A lock whose heartbeat
stopped long ago, or whose process died on this host, is removed under a
takeover guard file, so two processes cannot take over the same stale lock.
"""

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import threading
import time
import uuid

from tslab.core.meta import hostname
from tslab.core.paths import LOCK_FILE
from tslab.jobs.local import process_alive

__all__ = [
    'Heartbeat',
    'RunLockedError',
    'acquire_lock',
    'is_run_active',
    'read_lock_owner',
    'release_lock',
    'remove_stale_lock',
]

_LOGGER = logging.getLogger(__name__)
_HEARTBEAT_SECONDS = 60.0
_STALE_AFTER_SECONDS = 600.0
_TAKEOVER_GUARD_SECONDS = 30.0
_LOCK_ATTEMPTS = 50
_LOCK_RETRY_SECONDS = 0.05


class RunLockedError(RuntimeError):
    """Raised when another live process owns the run folder."""


@dataclass(frozen=True, kw_only=True)
class _LockOwner:
    """Contents of the ``.lock`` file.

    ``nonce`` is random per ``Run``: it lets a process check that the lock it is
    about to touch is still its own, even if its pid was reused or a stale lock
    was taken over.
    """

    pid: int
    host: str
    nonce: str | None = None


class Heartbeat:
    """Touch the lock file every ``_HEARTBEAT_SECONDS`` from a daemon thread.

    The modification time of the lock then tells any host whether its owner is
    still alive: a job killed by SLURM stops touching it. The thread stops
    touching the file once another process owns it.
    """

    def __init__(self, *, path: Path, nonce: str) -> None:
        """Start the daemon thread that refreshes the lock file.

        Args:
            path: The lock file to refresh.
            nonce: The owner nonce written into the lock file.
        """
        self._path = path
        self._nonce = nonce
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._beat, name='tslab-heartbeat', daemon=True)
        self._thread.start()

    def _beat(self) -> None:
        while not self._stopped.wait(timeout=_HEARTBEAT_SECONDS):
            owner = read_lock_owner(path=self._path)
            if owner is None or owner.nonce != self._nonce:
                _LOGGER.warning(
                    '%s is no longer owned by this run; stopped refreshing it', self._path
                )
                return
            try:
                os.utime(self._path)
            except OSError as error:
                _LOGGER.warning('could not refresh %s: %s', self._path.as_posix(), error)

    def stop(self) -> None:
        """Stop beating and wait for the thread to end."""
        self._stopped.set()
        self._thread.join()


def is_run_active(*, directory: Path) -> bool:
    """Return whether a live process holds the lock of a run folder.

    A lock is live when its heartbeat is recent and, on this host, its process
    still exists. A process of another host cannot be checked, so there only the
    heartbeat counts.

    Args:
        directory: The run folder.
    """
    path = directory / LOCK_FILE
    try:
        age = time.time() - path.stat().st_mtime
    except FileNotFoundError:
        return False
    return not _lock_is_stale(owner=read_lock_owner(path=path), age=age)


def _lock_is_stale(*, owner: _LockOwner | None, age: float) -> bool:
    if age > _STALE_AFTER_SECONDS:
        return True
    return owner is not None and owner.host == hostname() and not process_alive(pid=owner.pid)


def _retry_or_refuse(*, path: Path) -> None:
    """Return when the caller should look again; raise when a live process owns the lock."""
    owner = read_lock_owner(path=path)
    try:
        age = time.time() - path.stat().st_mtime
    except FileNotFoundError:
        return
    if not _lock_is_stale(owner=owner, age=age):
        message = (
            f'{path.parent.as_posix()} is in use by {owner or "an unknown process"}; '
            f'if no such process exists, delete {path.as_posix()}'
        )
        raise RunLockedError(message) from None
    if not remove_stale_lock(path=path):
        time.sleep(_LOCK_RETRY_SECONDS)  # another process is deciding; look again


def acquire_lock(*, path: Path) -> str:
    """Create the lock file atomically, replacing it once if it is stale.

    ``O_CREAT | O_EXCL`` makes creation atomic: exactly one process succeeds.
    A lock is stale when its heartbeat stopped more than ``_STALE_AFTER_SECONDS``
    ago (any host), or when its process is dead (this host). Removing a stale
    lock goes through ``remove_stale_lock``, so two processes cannot both remove
    it and then both create their own.

    Returns:
        The nonce stored in the lock; pass it to ``release_lock``.

    Raises:
        RunLockedError: If a live process owns the lock.
    """
    nonce = uuid.uuid4().hex
    payload = json.dumps({'pid': os.getpid(), 'host': hostname(), 'nonce': nonce})
    for _attempt in range(_LOCK_ATTEMPTS):
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            _retry_or_refuse(path=path)
            continue
        with os.fdopen(descriptor, mode='w', encoding='utf-8') as handle:
            handle.write(payload)
        return nonce
    message = f'could not acquire {path.as_posix()}'
    raise RunLockedError(message)


def remove_stale_lock(*, path: Path) -> bool:
    """Delete a lock that is still stale, while holding the takeover guard.

    The guard file serializes takeovers. Without it, two processes could both
    judge the same lock stale, one could delete it and create its own, and the
    other could then delete that fresh lock. Under the guard the staleness is
    checked again before the delete.

    Returns:
        ``False`` if another process holds the guard, ``True`` otherwise.
    """
    guard = path.with_name(f'{path.name}.takeover')
    try:
        descriptor = os.open(guard, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        _clear_old_guard(guard=guard)
        return False
    os.close(descriptor)
    try:
        _remove_if_stale(path=path)
    finally:
        guard.unlink(missing_ok=True)
    return True


def _clear_old_guard(*, guard: Path) -> None:
    try:
        if time.time() - guard.stat().st_mtime > _TAKEOVER_GUARD_SECONDS:
            guard.unlink(missing_ok=True)  # its holder died mid-takeover
    except FileNotFoundError:
        pass


def _remove_if_stale(*, path: Path) -> None:
    owner = read_lock_owner(path=path)
    try:
        age = time.time() - path.stat().st_mtime
    except FileNotFoundError:
        return
    if _lock_is_stale(owner=owner, age=age):
        _LOGGER.warning(
            'removing stale lock %s left by %s (last heartbeat %.0f s ago)',
            path.as_posix(),
            owner or 'an unknown process',
            age,
        )
        path.unlink(missing_ok=True)


def release_lock(*, path: Path, nonce: str) -> None:
    """Delete the lock file, but only if this run still owns it."""
    owner = read_lock_owner(path=path)
    if owner is not None and owner.nonce == nonce:
        path.unlink(missing_ok=True)
    elif path.exists():
        _LOGGER.warning('%s now belongs to another process; leaving it in place', path.as_posix())


def read_lock_owner(*, path: Path) -> _LockOwner | None:
    """Return the lock owner, or ``None`` if the file is unreadable or malformed."""
    try:
        document = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    if not isinstance(document, dict):
        return None
    pid, host, nonce = document.get('pid'), document.get('host'), document.get('nonce')
    if not isinstance(pid, int) or not isinstance(host, str):
        return None
    return _LockOwner(pid=pid, host=host, nonce=nonce if isinstance(nonce, str) else None)

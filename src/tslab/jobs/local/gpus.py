"""GPU lock files and GPU discovery for local machines."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import fcntl
import logging
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile

__all__ = ['GpuLease', 'busy_gpus', 'gpu_lock_folder', 'try_lease_gpu', 'visible_gpus']

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class GpuLease:
    """A held lock on one GPU, open in the given file descriptors.

    Attributes:
        gpu_id: The id of the GPU, as passed by the caller.
        fds: The open file descriptors that hold the lock; closing them
            releases it, also when the process exits.
    """

    gpu_id: str
    fds: tuple[int, ...]

    def release(self) -> None:
        """Close the file descriptors and release the lock."""
        for fd in self.fds:
            os.close(fd)


def _open_lock(*, folder: Path, name: str) -> int:
    """Open one lock file for writing, with an inheritable file descriptor."""
    fd = os.open(folder / name, os.O_RDWR | os.O_CREAT, 0o600)
    # why: the lock must survive exec, so the child keeps holding it with pass_fds
    os.set_inheritable(fd, True)  # noqa: FBT003 # why: os.set_inheritable takes positional arguments only
    return fd


def _lock_now(*, fd: int, lock: int) -> bool:
    """Try to take a flock without waiting; close the fd when it is held."""
    try:
        fcntl.flock(fd, lock | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return False
    return True


def gpu_lock_folder(*, name: str, environment: Mapping[str, str]) -> Path:
    """The shared folder for GPU lock files, created if it is missing.

    The folder is ``$XDG_RUNTIME_DIR/<name>`` when the variable is set, else
    ``<tempfile.gettempdir()>/<name>-<uid>`` so that other users never share
    the lock files.

    Args:
        name: The name of the folder, for example ``'gpus'``.
        environment: The environment to read ``XDG_RUNTIME_DIR`` from.

    Returns:
        The lock folder, created with mode 0700.

    Raises:
        PermissionError: If the folder is owned by another user.
    """
    # why: an empty XDG_RUNTIME_DIR means unset, per the XDG base directory spec
    runtime_dir = environment.get('XDG_RUNTIME_DIR') or None
    if runtime_dir is not None:
        folder = Path(runtime_dir) / name
    else:
        folder = Path(tempfile.gettempdir()) / f'{name}-{os.getuid()}'
    folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = folder.stat()
    if info.st_uid != os.getuid():
        message = (
            f'the GPU lock folder {folder} is owned by uid {info.st_uid}; '
            f'fix the owner of the folder or set XDG_RUNTIME_DIR to a folder owned by you'
        )
        raise PermissionError(message)
    return folder


def try_lease_gpu(*, folder: Path, gpu_ids: Sequence[str], tasks_per_gpu: int) -> GpuLease | None:
    """Try to take the lock of one GPU, in the order of the ids.

    With one task per GPU the lock is exclusive on ``gpu-<id>.lock``. With
    more tasks per GPU the lock is shared on ``gpu-<id>.lock`` plus an
    exclusive lock on the first free slot ``gpu-<id>.<k>.lock`` below
    ``tasks_per_gpu``.

    Args:
        folder: The lock folder, from ``gpu_lock_folder``.
        gpu_ids: The ids to try, in order.
        tasks_per_gpu: The number of tasks that may share one GPU.

    Returns:
        The lease, or ``None`` when every id is held.
    """
    for gpu_id in gpu_ids:
        main_fd = _open_lock(folder=folder, name=f'gpu-{gpu_id}.lock')
        main_lock = fcntl.LOCK_EX if tasks_per_gpu == 1 else fcntl.LOCK_SH
        if not _lock_now(fd=main_fd, lock=main_lock):
            continue
        if tasks_per_gpu == 1:
            return GpuLease(gpu_id=gpu_id, fds=(main_fd,))
        for slot in range(tasks_per_gpu):
            slot_fd = _open_lock(folder=folder, name=f'gpu-{gpu_id}.{slot}.lock')
            if _lock_now(fd=slot_fd, lock=fcntl.LOCK_EX):
                return GpuLease(gpu_id=gpu_id, fds=(main_fd, slot_fd))
        os.close(main_fd)
    return None


def _cuda_visible_devices(*, value: str) -> tuple[str, ...]:
    """The ids from a ``CUDA_VISIBLE_DEVICES`` value."""
    if value == '':
        return ()
    ids: list[str] = []
    for entry in value.split(','):
        try:
            int(entry)
        except ValueError as error:
            message = (
                f'non-integer GPU id in CUDA_VISIBLE_DEVICES: {entry!r}; list numeric ids in gpus'
            )
            raise ValueError(message) from error
        ids.append(entry)
    return tuple(ids)


def _nvidia_smi(*, arguments: Sequence[str]) -> str | None:
    """Run nvidia-smi with a 10 second timeout.

    Args:
        arguments: The arguments for nvidia-smi, without the command itself.

    Returns:
        The standard output, or ``None`` when the command is missing, fails,
        or times out. A failure logs one warning.
    """
    binary = shutil.which('nvidia-smi')
    if binary is None:
        return None
    try:
        completed = subprocess.run(  # noqa: S603 # why: a fixed binary name with fixed query arguments
            [binary, *arguments], capture_output=True, text=True, timeout=10, check=False
        )
    except subprocess.TimeoutExpired:
        _LOGGER.warning('nvidia-smi timed out after 10 seconds: %s', ' '.join(arguments))
        return None
    except OSError as error:
        _LOGGER.warning('nvidia-smi failed: %s', error)
        return None
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        _LOGGER.warning('nvidia-smi failed with exit code %s: %s', completed.returncode, stderr)
        return None
    return completed.stdout


def visible_gpus(*, environment: Mapping[str, str]) -> tuple[str, ...]:
    """The ids of the GPUs the caller may use, in discovery order.

    The rules, in order: a set ``CUDA_VISIBLE_DEVICES`` wins (an empty value
    means no GPU); else nvidia-smi lists the index of every GPU (a missing
    nvidia-smi logs one debug line and means no GPU; a failing or timed-out
    nvidia-smi logs one warning and means no GPU); else, on Darwin arm64, the
    Apple GPU ``'mps'``; else no GPU.

    Args:
        environment: The environment to read ``CUDA_VISIBLE_DEVICES`` from.

    Returns:
        The ids of the visible GPUs, or an empty tuple for none.

    Raises:
        ValueError: If ``CUDA_VISIBLE_DEVICES`` holds a non-integer id.
    """
    value = environment.get('CUDA_VISIBLE_DEVICES')
    if value is not None:
        return _cuda_visible_devices(value=value)
    if shutil.which('nvidia-smi') is None:
        _LOGGER.debug('nvidia-smi is missing; assuming no NVIDIA GPUs')
        return ('mps',) if platform.system() == 'Darwin' and platform.machine() == 'arm64' else ()
    output = _nvidia_smi(arguments=['--query-gpu=index', '--format=csv,noheader'])
    if output is None:
        return ()
    return tuple(line.strip() for line in output.splitlines() if line.strip())


def busy_gpus(*, gpu_ids: Sequence[str]) -> frozenset[str]:
    """The ids of the given GPUs that run a compute application.

    Args:
        gpu_ids: The ids of the GPUs to check.

    Returns:
        The subset of ``gpu_ids`` with a compute application on it. A failing
        nvidia-smi logs one warning and counts no GPU as busy.
    """
    apps = _nvidia_smi(arguments=['--query-compute-apps=gpu_uuid', '--format=csv,noheader'])
    if apps is None:
        return frozenset[str]()
    gpus = _nvidia_smi(arguments=['--query-gpu=index,uuid', '--format=csv,noheader'])
    if gpus is None:
        return frozenset[str]()
    index_by_uuid: dict[str, str] = {}
    for line in gpus.splitlines():
        index, _, uuid = line.partition(',')
        if uuid:
            index_by_uuid[uuid.strip()] = index.strip()
    busy = {
        index
        for line in apps.splitlines()
        if (index := index_by_uuid.get(line.strip())) is not None and index in gpu_ids
    }
    return frozenset(busy)

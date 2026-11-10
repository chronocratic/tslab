"""Run metadata written to ``meta.json`` at the run root.

``meta.json`` is a public, versioned format (``format_version``): it lets a
later study layer group runs (``config_hash_without_seed``) and lets a reader
reproduce a run (versions, git commit).
"""

from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess

__all__ = [
    'CODE_SNAPSHOT_VARIABLE',
    'CODE_SNAPSHOT_VARIABLES',
    'META_FORMAT_VERSION',
    'collect_code_snapshot',
    'collect_git_state',
    'collect_hardware',
    'collect_versions',
    'hostname',
    'package_version',
]

META_FORMAT_VERSION = 1
_GIT_TIMEOUT_SECONDS = 10
CODE_SNAPSHOT_VARIABLE = 'TSLAB_CODE_SNAPSHOT'
CODE_SNAPSHOT_VARIABLES = (
    CODE_SNAPSHOT_VARIABLE,
    'TSLAB_CODE_SNAPSHOT_HASH',
    'TSLAB_CODE_SNAPSHOT_COMMIT',
    'TSLAB_CODE_SNAPSHOT_DIRTY',
    'TSLAB_CODE_SNAPSHOT_UNTRACKED_COPIED',
)


def package_version(*, distribution: str) -> str | None:
    """Return the installed version of ``distribution``, or ``None`` if missing."""
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def collect_versions() -> dict[str, str | None]:
    """Return the versions that most affect results.

    Returns:
        Mapping with keys ``python``, ``tslab``, ``torch``, ``lightning``.
    """
    return {
        'python': platform.python_version(),
        'tslab': package_version(distribution='tslab'),
        'torch': package_version(distribution='torch'),
        'lightning': package_version(distribution='lightning'),
    }


def collect_hardware() -> dict[str, object]:
    """Return the CPU and CUDA hardware visible to this process.

    Returns:
        Mapping with keys ``cpu_count``, ``cuda_available``, ``cuda_version``,
        ``gpus`` and ``cuda_visible_devices``.
    """
    import torch  # noqa: PLC0415 # why: torch is slow to import and metadata is read by CLI paths

    cuda_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if cuda_available else 0
    return {
        'cpu_count': os.cpu_count() or 0,
        'cuda_available': cuda_available,
        'cuda_version': torch.version.cuda,
        'gpus': [torch.cuda.get_device_name(index) for index in range(gpu_count)],
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
    }


def collect_code_snapshot(*, environ: Mapping[str, str]) -> dict[str, object] | None:
    """Return the code snapshot block of ``meta.json`` from the job environment.

    The job launcher sets the ``TSLAB_CODE_SNAPSHOT*`` variables; a run without a
    code snapshot leaves them unset and gets ``None``. A flag that is not
    ``'true'`` or ``'false'`` gives ``None`` for that field: metadata must never
    stop a run.

    Args:
        environ: The environment of the job, usually ``os.environ``.

    Returns:
        Mapping with keys ``folder``, ``hash``, ``commit``, ``dirty`` and
        ``untracked_copied``, or ``None`` when the snapshot variable is unset
        or empty.
    """
    folder = environ.get(CODE_SNAPSHOT_VARIABLE, '')
    if not folder:
        return None
    return {
        'folder': folder,
        'hash': environ.get('TSLAB_CODE_SNAPSHOT_HASH', ''),
        'commit': environ.get('TSLAB_CODE_SNAPSHOT_COMMIT', ''),
        'dirty': _parse_flag(value=environ.get('TSLAB_CODE_SNAPSHOT_DIRTY')),
        'untracked_copied': _parse_flag(value=environ.get('TSLAB_CODE_SNAPSHOT_UNTRACKED_COPIED')),
    }


def _parse_flag(*, value: str | None) -> bool | None:
    if value == 'true':
        return True
    if value == 'false':
        return False
    return None


def collect_git_state(*, directory: Path) -> dict[str, object]:
    """Return the git commit and dirty flag of the repository containing ``directory``.

    Failures (git missing, not a repository, timeout) give ``None`` values
    instead of raising: metadata must never stop a run.

    Args:
        directory: Any folder inside the project of the user, usually the working directory.

    Returns:
        Mapping with keys ``commit`` (``str | None``) and ``dirty`` (``bool | None``).
    """
    git = shutil.which('git')
    if git is None:
        return {'commit': None, 'dirty': None}
    try:
        commit = _run_git(git=git, directory=directory, arguments=('rev-parse', 'HEAD'))
        porcelain = _run_git(git=git, directory=directory, arguments=('status', '--porcelain'))
    except (OSError, subprocess.SubprocessError):
        return {'commit': None, 'dirty': None}
    return {'commit': commit.strip(), 'dirty': bool(porcelain.strip())}


def _run_git(*, git: str, directory: Path, arguments: tuple[str, ...]) -> str:
    return subprocess.run(  # noqa: S603 # why: fixed git arguments, no user input
        [git, *arguments],
        cwd=directory,
        capture_output=True,
        text=True,
        check=True,
        timeout=_GIT_TIMEOUT_SECONDS,
    ).stdout


def hostname() -> str:
    """Return the host name of this machine."""
    return socket.gethostname()

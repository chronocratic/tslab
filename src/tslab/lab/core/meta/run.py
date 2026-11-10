"""Run metadata written to ``meta.json`` at the run root.

``meta.json`` is a public, versioned format (``format_version``): it lets a
later study layer group runs (``config_hash_without_seed``) and lets a reader
reproduce a run (versions, git commit).
"""

from collections.abc import Callable, Mapping
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
    'collectors_generation',
    'hostname',
    'package_version',
    'register_hardware',
    'register_versions',
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


_version_collectors: dict[str, Callable[[], Mapping[str, str | None]]] = {}
_hardware_collectors: dict[str, Callable[[], Mapping[str, object]]] = {}
_generation = 0


def collectors_generation() -> int:
    """A number that changes whenever a collector is added, to tell a cached result is stale."""
    return _generation


def register_versions(collector: Callable[[], Mapping[str, str | None]], *, name: str) -> None:
    """Add a function returning more package versions; the same ``name`` replaces the earlier one.

    Args:
        collector: Returns the extra version entries.
        name: What the collector reports on, for example ``jax``.
    """
    global _generation  # noqa: PLW0603 # why: counts the registrations of the process
    _version_collectors[name] = collector
    _generation += 1


def register_hardware(collector: Callable[[], Mapping[str, object]], *, name: str) -> None:
    """Add a function returning more hardware facts; the same ``name`` replaces the earlier one.

    Args:
        collector: Returns the extra hardware entries.
        name: What the collector reports on, for example ``jax``.
    """
    global _generation  # noqa: PLW0603 # why: counts the registrations of the process
    _hardware_collectors[name] = collector
    _generation += 1


def collect_versions() -> dict[str, str | None]:
    """Return the versions that most affect results.

    Returns:
        Mapping with keys ``python`` and ``tslab``, plus the keys of every
        registered collector.
    """
    versions: dict[str, str | None] = {
        'python': platform.python_version(),
        'tslab': package_version(distribution='tslab'),
    }
    for collector in _version_collectors.values():
        versions.update(collector())
    return versions


def collect_hardware() -> dict[str, object]:
    """Return the hardware visible to this process.

    Returns:
        Mapping with key ``cpu_count``, plus the keys of every registered collector.
    """
    hardware: dict[str, object] = {'cpu_count': os.cpu_count() or 0}
    for collector in _hardware_collectors.values():
        hardware.update(collector())
    return hardware


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

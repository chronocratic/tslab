"""Run metadata written to ``meta.json`` at the run root.

``meta.json`` is a public, versioned format (``format_version``): it lets a
later study layer group runs (``config_hash_without_seed``) and lets a reader
reproduce a run (versions, git commit).
"""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
import shutil
import socket
import subprocess

__all__ = [
    'META_FORMAT_VERSION',
    'collect_git_state',
    'collect_versions',
    'hostname',
    'package_version',
]

META_FORMAT_VERSION = 1
_GIT_TIMEOUT_SECONDS = 10


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


def collect_git_state(*, directory: Path) -> dict[str, object]:
    """Return the git commit and dirty flag of the repository containing ``directory``.

    Failures (git missing, not a repository, timeout) give ``None`` values
    instead of raising: metadata must never stop a run.

    Args:
        directory: Any folder inside the user's project, usually the working directory.

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
    """Return this machine's host name."""
    return socket.gethostname()

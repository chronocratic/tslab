"""The code snapshot: the copy of ``src/`` that submitted jobs run from.

The snapshot is content-addressed under
``<output_root>/.tslab-code-snapshots/<hash>``, so the same code is stored
once and later edits never change a queued or resubmitted run.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import logging
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, NoReturn

from tslab.core.meta import CODE_SNAPSHOT_VARIABLE, CODE_SNAPSHOT_VARIABLES
from tslab.core.paths import require_path

__all__ = [
    'CODE_SNAPSHOTS_FOLDER',
    'SRC_FOLDER',
    'CodeSnapshot',
    'CodeSnapshotError',
    'take_code_snapshot',
    'without_code_snapshot',
]

CODE_SNAPSHOTS_FOLDER = '.tslab-code-snapshots'
SRC_FOLDER = 'src'
_GIT_TIMEOUT_SECONDS = 10
_MAX_LISTED_UNTRACKED = 20

_LOGGER = logging.getLogger(__name__)


class CodeSnapshotError(ValueError):
    """A code snapshot cannot be taken; the message is shown as is."""


@dataclass(frozen=True, kw_only=True)
class CodeSnapshot:
    """The record of one code snapshot.

    Args:
        folder: The content-addressed folder holding the copy of ``src/``.
        hash: The first 16 hex characters of the SHA-256 over the copied files.
        commit: The ``HEAD`` of the project when the snapshot was taken.
        dirty: Whether a tracked file under ``src/`` differed from ``HEAD``.
        untracked_copied: Whether at least one untracked file was copied.
    """

    folder: Path
    hash: str  # why: matches the JSON key
    commit: str
    dirty: bool
    untracked_copied: bool

    def to_json(self) -> dict[str, object]:
        """The JSON form of the snapshot."""
        return {
            'folder': self.folder.as_posix(),
            'hash': self.hash,
            'commit': self.commit,
            'dirty': self.dirty,
            'untracked_copied': self.untracked_copied,
        }

    @classmethod
    def from_json(cls, *, document: Mapping[str, Any]) -> 'CodeSnapshot':
        """A snapshot from its JSON form."""
        return cls(
            folder=Path(document['folder']),
            hash=str(document['hash']),
            commit=str(document['commit']),
            dirty=bool(document['dirty']),
            untracked_copied=bool(document['untracked_copied']),
        )

    def variables(self) -> dict[str, str]:
        """The ``TSLAB_CODE_SNAPSHOT*`` environment variables of the job."""
        values = (
            self.folder.as_posix(),
            self.hash,
            self.commit,
            'true' if self.dirty else 'false',
            'true' if self.untracked_copied else 'false',
        )
        return dict(zip(CODE_SNAPSHOT_VARIABLES, values, strict=True))

    def pythonpath(self, *, current: str | None) -> str:
        """The ``PYTHONPATH`` that makes the job import the copy of ``src/``.

        Args:
            current: The ``PYTHONPATH`` of the caller, or ``None``.

        Returns:
            ``<folder>/src``, followed by ``:`` and the current value when
            the current value is non-empty.
        """
        source = str(self.folder / SRC_FOLDER)
        if current is None or not current:
            return source
        return f'{source}:{current}'


def without_code_snapshot(*, environ: Mapping[str, str]) -> dict[str, str]:
    """The environment without the code snapshot the caller runs from.

    A nested launch (a follow-up job, a local task) must not keep the
    snapshot of the launch that started it: the five variables go, and the
    ``<folder>/src`` entry leaves ``PYTHONPATH``, which is dropped when
    nothing is left.

    Args:
        environ: The environment to clean, usually ``os.environ``.

    Returns:
        A copy of the environment without the code snapshot entries.
    """
    cleaned = {
        name: value for name, value in environ.items() if name not in CODE_SNAPSHOT_VARIABLES
    }
    folder = environ.get(CODE_SNAPSHOT_VARIABLE, '')
    pythonpath = cleaned.get('PYTHONPATH')
    if not folder or pythonpath is None:
        return cleaned
    kept = [entry for entry in pythonpath.split(':') if entry != str(Path(folder) / SRC_FOLDER)]
    if not kept:
        cleaned.pop('PYTHONPATH')
        return cleaned
    cleaned['PYTHONPATH'] = ':'.join(kept)
    return cleaned


def take_code_snapshot(
    *,
    project_root: Path,
    output_root: Path,
    include_untracked: bool,
    check_untracked: bool,
    dry_run: bool,
) -> CodeSnapshot:
    """Copy ``src/`` of the project into a content-addressed code snapshot.

    Tracked files are copied with their working-tree bytes, so uncommitted
    edits are included; tracked files deleted from the working tree are
    skipped. Untracked files are copied when ``include_untracked`` or
    ``check_untracked`` is false; with untracked files present,
    ``check_untracked`` and not ``include_untracked`` stops before anything
    is written. Ignored files are never copied. The same code gives the
    same folder, which is reused, never overwritten.

    Args:
        project_root: The project folder holding ``src/``, inside a git repository.
        output_root: The output root the snapshot folder is created under.
        include_untracked: Copy untracked files under ``src/`` too.
        check_untracked: Stop when untracked files would be left out.
        dry_run: Run every check and compute the hash, copy nothing.

    Returns:
        The snapshot; the folder exists on disk unless ``dry_run`` is true.

    Raises:
        CodeSnapshotError: When the snapshot cannot be taken; the message is
            shown to the user as is and names the way out.
    """
    require_path(value=project_root, parameter='project_root')
    require_path(value=output_root, parameter='output_root')
    git = _check_git(project_root=project_root)
    commit = _head_commit(git=git, project_root=project_root)
    _check_src_layout(project_root=project_root)
    tracked = _tracked_files(git=git, project_root=project_root)
    untracked = _untracked_files(git=git, project_root=project_root)
    _stop_on_untracked(
        untracked=untracked, check_untracked=check_untracked, include_untracked=include_untracked
    )
    copies, untracked_copied = _copied_files(
        project_root=project_root,
        tracked=tracked,
        untracked=untracked,
        include_untracked=include_untracked,
        check_untracked=check_untracked,
    )
    dirty = _dirty_files(git=git, project_root=project_root)
    if dirty:
        _LOGGER.info('code snapshot includes uncommitted changes in %d files', len(dirty))
    digest = _hash_of_files(project_root=project_root, relative_paths=copies)
    folder = output_root / CODE_SNAPSHOTS_FOLDER / digest
    snapshot = CodeSnapshot(
        folder=folder,
        hash=digest,
        commit=commit,
        dirty=bool(dirty),
        untracked_copied=untracked_copied,
    )
    if dry_run:
        return snapshot
    if (folder / SRC_FOLDER).is_dir():
        return snapshot
    if folder.is_dir():
        # why: a folder that lost its src (e.g. deleted by hand) is not a
        # complete snapshot; recopy the code instead of reusing it
        _LOGGER.warning('code snapshot folder %s lost its src folder; recopying', folder)
        shutil.rmtree(folder)
    _copy_atomically(project_root=project_root, copies=copies, folder=folder, digest=digest)
    return snapshot


def _check_git(*, project_root: Path) -> str:
    """The git executable, or the error that names the way out."""
    git = shutil.which('git')
    if git is None:
        _stop_no_git(project_root=project_root)
    try:
        _run_git(git=git, project_root=project_root, arguments=('rev-parse', '--show-toplevel'))
    except (OSError, subprocess.SubprocessError):
        _stop_no_git(project_root=project_root)
    return git


def _stop_no_git(*, project_root: Path) -> NoReturn:
    message = (
        f'code snapshots need git: {project_root} is not in a git repository; '
        f'rerun with --no-code-snapshot (run the live code)'
    )
    raise CodeSnapshotError(message)


def _head_commit(*, git: str, project_root: Path) -> str:
    """The ``HEAD`` of the repository, or the error that names the way out."""
    try:
        return _run_git(git=git, project_root=project_root, arguments=('rev-parse', 'HEAD')).strip()
    except (OSError, subprocess.SubprocessError) as error:
        message = (
            f'code snapshots need git: the repository of {project_root} has no commit yet; '
            f'rerun with --no-code-snapshot (run the live code)'
        )
        raise CodeSnapshotError(message) from error


def _check_src_layout(*, project_root: Path) -> None:
    if (project_root / SRC_FOLDER).is_dir():
        return
    message = (
        f'code snapshots need the src layout: {project_root} has no src folder; '
        f'rerun with --no-code-snapshot (run the live code)'
    )
    raise CodeSnapshotError(message)


def _tracked_files(*, git: str, project_root: Path) -> list[str]:
    return _list_files(git=git, project_root=project_root, flags=('--cached',))


def _untracked_files(*, git: str, project_root: Path) -> list[str]:
    return _list_files(git=git, project_root=project_root, flags=('--others', '--exclude-standard'))


def _list_files(*, git: str, project_root: Path, flags: tuple[str, ...]) -> list[str]:
    raw = _run_git(
        git=git,
        project_root=project_root,
        arguments=('ls-files', '-z', *flags, '--', f'{SRC_FOLDER}/'),
    )
    return [path for path in raw.split('\0') if path]


def _stop_on_untracked(
    *, untracked: list[str], check_untracked: bool, include_untracked: bool
) -> None:
    if not untracked or not check_untracked or include_untracked:
        return
    lines = [
        (
            f'{len(untracked)} files under src/ are not tracked by git; '
            f'the snapshot would leave them out:'
        )
    ]
    lines.extend(f'  {path}' for path in untracked[:_MAX_LISTED_UNTRACKED])
    hidden = len(untracked) - _MAX_LISTED_UNTRACKED
    if hidden > 0:
        lines.append(f'  and {hidden} more')
    lines.append(
        'Run `git add` on them, or rerun with --code-snapshot-include-untracked (copy them too)'
    )
    lines.append('or --no-code-snapshot (run the live code).')
    message = '\n'.join(lines)
    raise CodeSnapshotError(message)


def _copied_files(
    *,
    project_root: Path,
    tracked: list[str],
    untracked: list[str],
    include_untracked: bool,
    check_untracked: bool,
) -> tuple[list[str], bool]:
    copies = [path for path in tracked if (project_root / path).is_file()]
    untracked_copied = False
    if untracked and (include_untracked or not check_untracked):
        copies.extend(untracked)
        untracked_copied = True
    return copies, untracked_copied


def _dirty_files(*, git: str, project_root: Path) -> list[str]:
    raw = _run_git(
        git=git,
        project_root=project_root,
        arguments=('diff', '--name-only', '-z', 'HEAD', '--', f'{SRC_FOLDER}/'),
    )
    return [path for path in raw.split('\0') if path]


def _hash_of_files(*, project_root: Path, relative_paths: list[str]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(relative_paths):
        data = (project_root / relative).read_bytes()
        digest.update(relative.encode('utf-8') + b'\0' + len(data).to_bytes(8, 'big') + data)
    return digest.hexdigest()[:16]


def _copy_atomically(*, project_root: Path, copies: list[str], folder: Path, digest: str) -> None:
    """Copy into a temporary sibling folder, then rename to ``folder``.

    A crash or a folder that appears meanwhile must never leave a partial
    copy at ``folder``, and the published bytes must be the bytes the
    ``digest`` was computed from.
    """
    folder.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=folder.parent, prefix=f'.{folder.name}-'))
    try:
        _copy_files(project_root=project_root, copies=copies, destination=temporary)
        _check_copy_digest(temporary=temporary, copies=copies, digest=digest)
        _publish_copy(temporary=temporary, folder=folder)
    except CodeSnapshotError:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    except (OSError, shutil.Error) as error:
        shutil.rmtree(temporary, ignore_errors=True)
        message = f'cannot copy the code snapshot to {folder}: {error}'
        raise CodeSnapshotError(message) from error


def _check_copy_digest(*, temporary: Path, copies: list[str], digest: str) -> None:
    """The copied files match the recorded hash, or the error that names the way out.

    The hash was computed from the project before the copy; an edit in
    between would publish different bytes under the recorded hash and break
    the content addressing.
    """
    copied = _hash_of_files(project_root=temporary, relative_paths=copies)
    if copied == digest:
        return
    message = (
        'the code changed while the code snapshot was being copied; '
        'rerun the command to take a new snapshot'
    )
    raise CodeSnapshotError(message)


def _copy_files(*, project_root: Path, copies: list[str], destination: Path) -> None:
    for relative in copies:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project_root / relative, target)


def _publish_copy(*, temporary: Path, folder: Path) -> None:
    try:
        temporary.rename(folder)
    except OSError:
        if not folder.is_dir():
            raise
        shutil.rmtree(temporary)


def _run_git(*, git: str, project_root: Path, arguments: tuple[str, ...]) -> str:
    return subprocess.run(  # noqa: S603 # why: fixed git arguments, no user input
        [git, *arguments],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
        timeout=_GIT_TIMEOUT_SECONDS,
    ).stdout

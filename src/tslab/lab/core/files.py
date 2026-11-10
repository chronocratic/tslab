"""Atomic file writing and JSON file reading.

Files are written to a temporary sibling, flushed to disk and renamed into
place, so a crash can never leave a half-written file that looks valid.
"""

import json
import os
from pathlib import Path
import shutil
import threading
from typing import cast
import uuid

from tslab.core.paths import _TEMPORARY_MARKER, require_path

__all__ = [
    'JsonFileError',
    'atomic_write_bytes',
    'atomic_write_json',
    'atomic_write_text',
    'read_json_object',
    'replace_directory',
]


class JsonFileError(ValueError):
    """Raised when a JSON file written by TSLab is unreadable or has the wrong shape."""


def atomic_write_text(*, path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically.

    Writes to a temporary sibling file, flushes it to disk with ``fsync`` and
    renames it over the target. A rename within one directory is atomic, so a
    reader sees either the old file or the new one, never a partial file. The
    temporary name is unique per call, so threads and processes never share it,
    and the file is removed again if writing fails.

    Args:
        path: Destination file.
        text: Content to write (UTF-8).
    """
    _atomic_write(path=path, data=text.encode('utf-8'))


def atomic_write_bytes(*, path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically, like :func:`atomic_write_text`.

    Args:
        path: Destination file.
        data: Content to write.
    """
    _atomic_write(path=path, data=data)


def _atomic_write(*, path: Path, data: bytes) -> None:
    require_path(value=path, parameter='path')
    path.parent.mkdir(parents=True, exist_ok=True)
    unique = f'{os.getpid()}-{threading.get_ident()}-{uuid.uuid4().hex[:8]}'
    temporary_path = path.with_name(f'{path.name}{_TEMPORARY_MARKER}{unique}')
    try:
        with temporary_path.open(mode='wb') as file_handle:
            file_handle.write(data)
            file_handle.flush()
            os.fsync(file_handle.fileno())
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    _sync_directory(directory=path.parent)


def atomic_write_json(*, path: Path, document: object) -> None:
    """Write ``document`` as indented JSON with sorted keys, atomically.

    Args:
        path: Destination file.
        document: A JSON-compatible value.
    """
    atomic_write_text(path=path, text=json.dumps(document, indent=2, sort_keys=True) + '\n')


def read_json_object(*, path: Path) -> dict[str, object]:
    """Read a JSON file that must hold an object.

    Args:
        path: The file.

    Returns:
        The parsed object.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        JsonFileError: If the file is not valid UTF-8 JSON or not a JSON object;
            the message names the file.
    """
    require_path(value=path, parameter='path')
    try:
        document = json.loads(path.read_text(encoding='utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        message = f'{path.as_posix()} is not valid JSON: {error}'
        raise JsonFileError(message) from error
    if not isinstance(document, dict):
        message = f'{path.as_posix()} must be a JSON object, got {type(document).__name__}'
        raise JsonFileError(message)
    return cast('dict[str, object]', document)


def _sync_directory(*, directory: Path) -> None:
    """Flush a directory entry to disk so a rename survives a crash (best effort)."""
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return  # not supported here (for example on Windows)
    try:
        os.fsync(descriptor)
    except OSError:
        pass  # some network filesystems refuse fsync on a directory
    finally:
        os.close(descriptor)


def replace_directory(*, source: Path, destination: Path) -> None:
    """Move a fully written ``source`` directory to ``destination``.

    An existing ``destination`` is first renamed aside, then ``source`` takes its
    place, then the old folder is deleted. The folder is therefore missing only
    between two renames, and the old result comes back if the second rename
    fails. ``source`` must be on the same filesystem (TSLab always creates it as
    a sibling), so each rename is cheap and atomic.

    Args:
        source: Completely written temporary directory.
        destination: Final location.
    """
    require_path(value=source, parameter='source')
    require_path(value=destination, parameter='destination')
    destination.parent.mkdir(parents=True, exist_ok=True)
    previous: Path | None = None
    if destination.exists():
        previous = destination.with_name(
            f'{destination.name}{_TEMPORARY_MARKER}old-{uuid.uuid4().hex[:8]}'
        )
        destination.rename(previous)
    try:
        source.rename(destination)
    except BaseException:
        if previous is not None:
            previous.rename(destination)
        raise
    if previous is not None:
        shutil.rmtree(previous, ignore_errors=True)

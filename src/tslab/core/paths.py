"""Path validation and atomic file writing.

Every public TSLab function that takes a filesystem location requires a
``pathlib.Path`` (never a ``str``). The helpers here enforce that rule, validate
the names TSLab turns into folders, and write files atomically so a crash can
never leave a half-written file that looks valid.
"""

import os
from pathlib import Path
import re
import shutil

__all__ = [
    'DEFAULT_INSTANCE_NAME',
    'NAME_PART_PATTERN',
    'RESERVED_STAGE_NAMES',
    'InstanceName',
    'atomic_write_text',
    'normalize_instance_name',
    'replace_directory',
    'require_path',
    'validate_name_part',
    'validate_relative_path',
    'validate_stage_name',
]

type InstanceName = str | tuple[str, ...]
"""An instance name: one part (``"smooth"``) or several (``("forecasting", "h24", "smooth")``)."""

NAME_PART_PATTERN = re.compile(r'[A-Za-z0-9._-]+')
DEFAULT_INSTANCE_NAME = 'default'
RESERVED_STAGE_NAMES = frozenset({'logs', 'trackers'})
_FORBIDDEN_PARTS = frozenset({'.', '..'})


def require_path(*, value: object, parameter: str) -> Path:
    """Return ``value`` if it is a ``pathlib.Path``, otherwise raise ``TypeError``.

    Strings are rejected on purpose: TSLab treats paths as typed values so that
    path handling is identical on every operating system.

    Args:
        value: The object received by a public function.
        parameter: Name of the parameter, used in the error message.

    Returns:
        The same object, typed as ``Path``.

    Raises:
        TypeError: If ``value`` is not a ``Path``.
    """
    if not isinstance(value, Path):
        message = (
            f'{parameter} must be pathlib.Path, got {type(value).__name__}; wrap it with Path(...)'
        )
        raise TypeError(message)
    return value


def validate_name_part(*, part: str) -> str:
    """Validate one component of an instance name or run path.

    Args:
        part: A single name component, for example ``'smooth'`` or ``'h24'``.

    Returns:
        The same component.

    Raises:
        TypeError: If ``part`` is not a ``str``.
        ValueError: If ``part`` is ``'.'``, ``'..'`` or contains characters
            outside ``[A-Za-z0-9._-]``.
    """
    if not isinstance(part, str):
        message = f'name parts must be str, got {type(part).__name__}'
        raise TypeError(message)
    if part in _FORBIDDEN_PARTS or NAME_PART_PATTERN.fullmatch(part) is None:
        message = (
            f'invalid name part {part!r}: use only letters, digits, ".", "_" and "-", '
            f'and not "." or ".."'
        )
        raise ValueError(message)
    return part


def validate_stage_name(*, name: str) -> str:
    """Validate a stage name, which becomes a top-level folder of the run.

    Args:
        name: The stage name, for example ``'encode'``.

    Returns:
        The same name.

    Raises:
        TypeError: If ``name`` is not a ``str``.
        ValueError: If ``name`` is not a valid name part or collides with a
            folder TSLab itself creates in the run folder.
    """
    validate_name_part(part=name)
    if name in RESERVED_STAGE_NAMES:
        message = (
            f'stage name {name!r} is reserved by TSLab; reserved: {sorted(RESERVED_STAGE_NAMES)}'
        )
        raise ValueError(message)
    return name


def normalize_instance_name(*, name: InstanceName | None) -> tuple[str, ...]:
    """Turn an instance name into validated folder parts.

    Args:
        name: ``None`` (meaning ``"default"``), one part, or a tuple of parts.

    Returns:
        The parts as a non-empty tuple.

    Raises:
        TypeError: If ``name`` is neither ``str``, ``tuple`` nor ``None``.
        ValueError: If the tuple is empty or a part is invalid.
    """
    if name is None:
        return (DEFAULT_INSTANCE_NAME,)
    if isinstance(name, str):
        return (validate_name_part(part=name),)
    if not isinstance(name, tuple):
        message = f'name must be str or tuple[str, ...], got {type(name).__name__}'
        raise TypeError(message)
    if not name:
        message = 'name must not be an empty tuple'
        raise ValueError(message)
    return tuple(validate_name_part(part=part) for part in name)


def validate_relative_path(*, path: Path, parameter: str) -> Path:
    """Validate a relative path used to place a run under its root folder.

    Args:
        path: The candidate relative path.
        parameter: Name of the parameter, used in error messages.

    Returns:
        The same path.

    Raises:
        TypeError: If ``path`` is not a ``Path``.
        ValueError: If ``path`` is absolute, empty, or has an invalid part.
    """
    require_path(value=path, parameter=parameter)
    if path.is_absolute():
        message = f'{parameter} must be relative, got absolute path {path.as_posix()!r}'
        raise ValueError(message)
    if not path.parts:
        message = f'{parameter} must not be empty'
        raise ValueError(message)
    for part in path.parts:
        validate_name_part(part=part)
    return path


def atomic_write_text(*, path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically.

    Writes to a temporary sibling file, flushes it to disk with ``fsync`` and
    renames it over the target. A rename within one directory is atomic, so a
    reader sees either the old file or the new one, never a partial file.

    Args:
        path: Destination file.
        text: Content to write (UTF-8).
    """
    require_path(value=path, parameter='path')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f'{path.name}.tmp-{os.getpid()}')
    with temporary_path.open(mode='w', encoding='utf-8') as file_handle:
        file_handle.write(text)
        file_handle.flush()
        os.fsync(file_handle.fileno())
    temporary_path.replace(path)


def replace_directory(*, source: Path, destination: Path) -> None:
    """Move a fully written ``source`` directory to ``destination``.

    Any existing ``destination`` is removed first. ``source`` must be on the same
    filesystem (TSLab always creates it as a sibling), so the final rename is
    cheap and atomic.

    Args:
        source: Completely written temporary directory.
        destination: Final location.
    """
    require_path(value=source, parameter='source')
    require_path(value=destination, parameter='destination')
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    source.rename(destination)

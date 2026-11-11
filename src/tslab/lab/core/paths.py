"""Path validation.

Every public TSLab function that takes a filesystem location requires a
``pathlib.Path`` (never a ``str``). The helpers here enforce that rule and
validate the names TSLab turns into folders.
"""

from pathlib import Path
import re

__all__ = [
    'CONFIG_FILE',
    'DEFAULT_INSTANCE_NAME',
    'ERROR_FILE',
    'LOCK_FILE',
    'LOGS_DIRECTORY',
    'META_FILE',
    'NAME_PART_PATTERN',
    'RESERVED_STAGE_NAMES',
    'STATUS_FILE',
    'TEMPORARY_MARKER',
    'TRACKERS_DIRECTORY',
    'InstanceName',
    'normalize_instance_name',
    'require_path',
    'validate_name_part',
    'validate_relative_path',
    'validate_stage_name',
]

type InstanceName = str | tuple[str, ...]
"""An instance name: one part (``"fgsm"``) or several (``("forecasting", "h24", "fgsm")``)."""

NAME_PART_PATTERN = re.compile(r'[A-Za-z0-9._-]+')
DEFAULT_INSTANCE_NAME = 'default'

# Names of the files and folders TSLab writes in a run folder. Readers elsewhere
# (the study layer, the status command) use these instead of spelling them again.
STATUS_FILE = 'status.json'
META_FILE = 'meta.json'
CONFIG_FILE = 'config.yaml'
LOCK_FILE = '.lock'
ERROR_FILE = 'error.txt'
LOGS_DIRECTORY = 'logs'
TRACKERS_DIRECTORY = 'trackers'

# A stage folder must not collide with anything TSLab writes next to it.
RESERVED_STAGE_NAMES = frozenset(
    {LOGS_DIRECTORY, TRACKERS_DIRECTORY, STATUS_FILE, META_FILE, CONFIG_FILE, LOCK_FILE}
)
_FORBIDDEN_PARTS = frozenset({'.', '..'})
# Marks the temporary siblings of an instance folder; a real name containing it
# could be deleted as a leftover.
TEMPORARY_MARKER = '.tmp-'


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
        part: A single name component, for example ``'fgsm'`` or ``'h24'``.

    Returns:
        The same component.

    Raises:
        TypeError: If ``part`` is not a ``str``.
        ValueError: If ``part`` is ``'.'``, ``'..'``, starts with a dot, contains
            ``'.tmp-'`` or contains characters outside ``[A-Za-z0-9._-]``.
    """
    if not isinstance(part, str):
        message = f'name parts must be str, got {type(part).__name__}'
        raise TypeError(message)
    if (
        part in _FORBIDDEN_PARTS
        or NAME_PART_PATTERN.fullmatch(part) is None
        or part.startswith('.')
        or TEMPORARY_MARKER in part
    ):
        message = (
            f'invalid name part {part!r}: use only letters, digits, ".", "_" and "-", '
            f'not starting with "." and not containing ".tmp-"'
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

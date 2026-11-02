"""The project settings: the committed ``tslab.toml`` at the project root.

The file holds the studies module, a host-to-machine table and the default
runs folder, so the study commands take short names and no ``--machine``.
It is found by walking up from the working folder; without one, the
built-in defaults hold and nothing errors.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import fnmatch
from pathlib import Path
import tomllib

from tslab.core.text import unknown_key_message

__all__ = [
    'PROJECT_FILE',
    'ProjectSettings',
    'expand_study',
    'find_project_file',
    'load_project_settings',
    'machine_for_host',
]

#: The file name the loader looks for, from the working folder upward.
PROJECT_FILE = 'tslab.toml'

_VALID_KEYS = (
    'studies',
    'machine_by_host',
    'runs',
    'code_snapshot',
    'code_snapshot_check_untracked',
)


@dataclass(frozen=True, kw_only=True)
class ProjectSettings:
    """The settings of one ``tslab.toml``; the built-in defaults without one.

    Attributes:
        path: The absolute path of the file in use; ``None`` without one.
        studies: The studies module, e.g. ``my_paper.studies``; short study
            names expand against it.
        machine_by_host: The host pattern to machine name pairs, in file
            order; the first pattern that matches wins.
        runs: The default runs folder of ``tslab status``, absolute and
            resolved against the folder holding the file.
        code_snapshot: Whether ``submit`` takes a code snapshot without
            ``--code-snapshot``.
        code_snapshot_check_untracked: Whether a code snapshot stops when
            untracked files under ``src/`` would be left out.
        file_keys: The top-level keys of the file, so a key holding an
            empty value still counts as present; empty without a file.
    """

    path: Path | None
    studies: str | None = None
    machine_by_host: tuple[tuple[str, str], ...] = ()
    runs: Path | None = None
    code_snapshot: bool = False
    code_snapshot_check_untracked: bool = True
    file_keys: frozenset[str] = frozenset[str]()


def find_project_file(*, start: Path) -> Path | None:
    """The nearest ``tslab.toml`` at ``start`` or in a parent folder.

    Args:
        start: The folder to start the walk, usually the working folder.

    Returns:
        The first file named ``tslab.toml`` walking up from ``start``, or
        ``None`` when no parent holds one.
    """
    folder = start
    while True:
        candidate = folder / PROJECT_FILE
        if candidate.is_file():
            return candidate
        if folder.parent == folder:
            return None
        folder = folder.parent


def load_project_settings(*, start: Path) -> ProjectSettings:
    """Read the nearest ``tslab.toml`` into settings; defaults without one.

    Args:
        start: The folder to start the search, usually the working folder.

    Returns:
        The settings of the nearest file, or the built-in defaults when no
        ``tslab.toml`` is found.

    Raises:
        ValueError: If the file cannot be read, does not parse, holds an
            unknown top-level key, or holds a value of the wrong type; the
            message starts with the path of the file.
    """
    path = find_project_file(start=start)
    if path is None:
        return ProjectSettings(path=None)
    try:
        table = tomllib.loads(path.read_text(encoding='utf-8'))
    except OSError as error:
        message = f'{path}: cannot read the file: {error}'
        raise ValueError(message) from error
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as error:
        message = f'{path}: the TOML does not parse: {error}'
        raise ValueError(message) from error
    return _settings_of(path=path, table=table)


def expand_study(*, reference: str, settings: ProjectSettings) -> str:
    """The full ``module:attribute`` reference behind a typed study name.

    Args:
        reference: The study as typed, short or full.
        settings: The project settings to expand against.

    Returns:
        ``reference`` unchanged when it holds a ``:`` or the settings have no
        studies module; otherwise the short name under that module.
    """
    if ':' in reference or settings.studies is None:
        return reference
    return f'{settings.studies}:{reference}'


def machine_for_host(*, settings: ProjectSettings, host: str) -> tuple[str, str] | None:
    """The machine of one host, from the first matching pattern in file order.

    Args:
        settings: The project settings holding the table.
        host: The name of the host the command runs on.

    Returns:
        The first ``(pattern, machine)`` whose pattern matches the host with
        a case-sensitive glob, or ``None`` when no pattern matches.
    """
    for pattern, machine in settings.machine_by_host:
        if fnmatch.fnmatchcase(host, pattern):
            return (pattern, machine)
    return None


def _settings_of(*, path: Path, table: Mapping[str, object]) -> ProjectSettings:
    unknown = [name for name in table if name not in _VALID_KEYS]
    if unknown:
        problem = unknown_key_message(location='', key=unknown[0], valid=_VALID_KEYS)
        message = f'{path}: {problem}'
        raise ValueError(message)
    return ProjectSettings(
        path=path,
        studies=_string_of(table=table, name='studies', path=path),
        machine_by_host=_machine_table(table=table, path=path),
        runs=_runs_of(table=table, path=path),
        code_snapshot=_bool_of(table=table, name='code_snapshot', path=path, default=False),
        code_snapshot_check_untracked=_bool_of(
            table=table, name='code_snapshot_check_untracked', path=path, default=True
        ),
        file_keys=frozenset(table),
    )


def _string_of(*, table: Mapping[str, object], name: str, path: Path) -> str | None:
    value = table.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        message = f'{path}: {name} must be a non-empty string, got {value!r}'
        raise ValueError(message)
    return value


def _bool_of(*, table: Mapping[str, object], name: str, path: Path, default: bool) -> bool:
    value = table.get(name)
    if value is None:
        return default
    if not isinstance(value, bool):
        message = f'{path}: {name} must be a boolean, got {value!r}'
        raise ValueError(message)  # noqa: TRY004 # why: bad file content, not arguments
    return value


def _runs_of(*, table: Mapping[str, object], path: Path) -> Path | None:
    value = _string_of(table=table, name='runs', path=path)
    if value is None:
        return None
    # why: a relative runs folder is relative to the file, not the working folder
    return path.parent / value


def _machine_table(*, table: Mapping[str, object], path: Path) -> tuple[tuple[str, str], ...]:
    value = table.get('machine_by_host')
    if value is None:
        return ()
    if not isinstance(value, Mapping):
        message = f'{path}: machine_by_host must be a table of host pattern to machine name'
        raise ValueError(message)  # noqa: TRY004 # why: bad file content, not arguments
    pairs: list[tuple[str, str]] = []
    for pattern, machine in value.items():
        if not isinstance(machine, str) or not machine:
            message = (
                f'{path}: machine_by_host[{pattern!r}] must be a non-empty string, got {machine!r}'
            )
            raise ValueError(message)  # why: bad file content, not arguments
        # why: tomllib keeps table order, and the first matching pattern must win
        pairs.append((pattern, machine))
    return tuple(pairs)

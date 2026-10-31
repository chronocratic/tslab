"""The register of machine names and the lookup of a ``--machine`` value."""

from collections.abc import Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import tomllib
from typing import cast

from tslab.core.paths import atomic_write_text
from tslab.study.machine.tables import MachineProfileError

__all__ = [
    'MachineLocation',
    'read_register',
    'register_file',
    'resolve_machine',
    'validate_machine_name',
    'write_register',
]

_REGISTER_VARIABLE = 'TSLAB_MACHINES_FILE'
_NAME_PATTERN = re.compile(r'[A-Za-z0-9_-]+')


def register_file() -> Path:
    """The register of machine names: ``tslab machine add`` writes it.

    Returns:
        ``$TSLAB_MACHINES_FILE`` when set, else ``$XDG_CONFIG_HOME/tslab/machines.toml``,
        else ``~/.config/tslab/machines.toml``.
    """
    override = os.environ.get(_REGISTER_VARIABLE, '')
    if override != '':
        return Path(override)
    config = os.environ.get('XDG_CONFIG_HOME', '')
    base = Path(config) if config != '' else Path.home() / '.config'
    return base / 'tslab' / 'machines.toml'


def validate_machine_name(*, name: str) -> str:
    """Check that ``name`` is a plain stem: letters, digits, ``_`` and ``-``.

    Args:
        name: The name to register.

    Returns:
        ``name``, unchanged.

    Raises:
        MachineProfileError: If ``name`` is empty, has a separator, a dot or a space.
    """
    if not _NAME_PATTERN.fullmatch(name):
        message = (
            f'machine name {name!r} must be a plain stem of letters, digits, "_" and "-" '
            '(no path separator, no .toml); example: cluster'
        )
        raise MachineProfileError(message)
    return name


def read_register() -> dict[str, Path]:
    """The registered machine names and their profile files.

    Returns:
        The name to absolute path mapping; empty when the register does not exist.

    Raises:
        MachineProfileError: If the register is not a table of name to path strings.
    """
    path = register_file()
    if not path.is_file():
        return {}
    try:
        table = tomllib.loads(path.read_text(encoding='utf-8'))
    except tomllib.TOMLDecodeError as error:
        message = f'machine register {path} is not valid TOML: {error}; fix or delete the file'
        raise MachineProfileError(message) from error
    if not all(isinstance(value, str) for value in table.values()):
        message = f'machine register {path} must map names to path strings; fix or delete the file'
        raise MachineProfileError(message)
    return {name: Path(cast('str', value)) for name, value in table.items()}


def write_register(*, entries: Mapping[str, Path]) -> None:
    """Replace the register with ``entries``, atomically.

    Args:
        entries: The name to absolute path mapping to store.
    """
    path = register_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f'{name} = {json.dumps(file.as_posix())}\n' for name, file in sorted(entries.items())]
    atomic_write_text(path=path, text=''.join(lines))


@dataclass(frozen=True, kw_only=True)
class MachineLocation:
    """The profile file a ``--machine`` value names.

    Attributes:
        path: The absolute path of the profile file.
        project_root: The folder the jobs run in: the folder that holds
            ``machines/`` when the name was found there, else the start folder.
    """

    path: Path
    project_root: Path


def resolve_machine(*, machine: str, start: Path) -> MachineLocation:
    """Find the profile file a ``--machine`` value names.

    A value with a path separator, a ``.toml`` suffix or that is an existing
    file is a path. Any other value is a name, looked up in this order:
    ``machines/<name>.toml`` in ``start`` and each folder above it (the first
    hit wins), then the register of ``tslab machine add``.

    Args:
        machine: The name or the path given on the command line.
        start: The folder to search from, usually the working directory.

    Returns:
        The absolute profile path and the folder the jobs run in.

    Raises:
        MachineProfileError: If a name is neither found above ``start`` nor
            registered, or is registered with a file that is gone.
    """
    if '/' in machine or '\\' in machine or machine.endswith('.toml') or Path(machine).is_file():
        return MachineLocation(path=Path(machine).resolve(), project_root=start)
    folders = [folder / 'machines' for folder in (start, *start.parents)]
    for folder in folders:
        if (folder / f'{machine}.toml').is_file():
            return MachineLocation(path=folder / f'{machine}.toml', project_root=folder.parent)
    registered = read_register()
    if machine in registered:
        path = registered[machine]
        if not path.is_file():
            message = (
                f'machine {machine!r} is registered with {path}, which does not exist; '
                f'register the new location with: tslab machine add {machine} <path>, '
                f'or drop it with: tslab machine remove {machine}'
            )
            raise MachineProfileError(message)
        return MachineLocation(path=path, project_root=start)
    message = (
        f'no machine named {machine!r}: no machines/{machine}.toml in {start} or a folder '
        f'above it, and the registered names are: {", ".join(sorted(registered)) or "none"}. '
        f'Pass a profile path, or register one with: tslab machine add {machine} <path>'
    )
    raise MachineProfileError(message)

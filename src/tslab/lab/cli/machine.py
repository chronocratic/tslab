"""The ``tslab machine`` subcommands: the register of machine names.

``--machine cluster`` finds ``machines/cluster.toml`` above the working
directory, and otherwise looks the name up in the register these commands edit.
"""

from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Annotated

import tyro

from tslab.lab.cli.common import print_json, UsageError
from tslab.lab.core.text import log_table
from tslab.lab.study.machine import (
    load_machine,
    MachineProfileError,
    read_register,
    validate_machine_name,
    write_register,
)

__all__ = ['MACHINE_COMMANDS', 'MachineAdd', 'MachineList', 'MachineRemove']

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class MachineAdd:
    """Register a machine profile under a name, so --machine <name> finds it anywhere.

    For example:
        tslab machine add local machines/local.toml

    Attributes:
        name: The name to register: letters, digits, _ and -.
        path: The machine profile file; stored as an absolute path.
    """

    name: Annotated[str, tyro.conf.arg(metavar='NAME'), tyro.conf.Positional]
    path: Annotated[Path, tyro.conf.arg(metavar='PROFILE'), tyro.conf.Positional]

    def run(self) -> int:
        """Validate the profile, store it and return the exit code."""
        try:
            validate_machine_name(name=self.name)
            file = self.path.expanduser().resolve()
            if not file.is_file():
                message = f'machine profile {file} does not exist; pass the path of a TOML file'
                raise MachineProfileError(message)  # noqa: TRY301 # why: one handler for all checks
            load_machine(path=file, lenient_env=True, environment=os.environ)
            write_register(
                entries={**read_register(environment=os.environ), self.name: file},
                environment=os.environ,
            )
        except MachineProfileError as error:
            raise UsageError(str(error)) from error
        _LOGGER.info('registered %s -> %s', self.name, file.as_posix())
        return 0


@dataclass(frozen=True, kw_only=True)
class MachineList:
    """Show the machine names --machine accepts here, and their profile files.

    For example:
        tslab machine list

    Attributes:
        json: Print one machine-list JSON document.
    """

    json: bool = False

    def _list_none(self) -> int:
        _LOGGER.info('no machines: none in machines/ above this folder, none registered')
        _LOGGER.info('register one: tslab machine add <name> <profile.toml>')
        if self.json:
            # When API is stable: version this id (tslab.machine-list/1); bump on a breaking change.
            print_json(document={'schema': 'tslab.machine-list', 'machines': []})
        return 0

    def run(self) -> int:
        """Print a table of names and return the exit code."""
        try:
            registered = read_register(environment=os.environ)
        except MachineProfileError as error:
            raise UsageError(str(error)) from error
        project = _project_machines(start=Path.cwd())
        if not project and not registered:
            return self._list_none()
        if self.json:
            machines = [
                {
                    'name': name,
                    'source': 'project',
                    'profile': file.resolve().as_posix(),
                    'note': '',
                }
                for name, file in project.items()
            ]
            machines.extend(
                {
                    'name': name,
                    'source': 'registered',
                    'profile': file.resolve().as_posix(),
                    'note': _register_note(name=name, file=file, project=project),
                }
                for name, file in sorted(registered.items())
            )
            # When API is stable: version this id (tslab.machine-list/1); bump on a breaking change.
            print_json(document={'schema': 'tslab.machine-list', 'machines': machines})
            _LOGGER.info('use one: tslab study submit STUDY --machine <name>')
            return 0
        rows = [
            (name, 'project', file.relative_to(Path.cwd(), walk_up=True).as_posix(), '')
            for name, file in project.items()
        ]
        rows += [
            (
                name,
                'registered',
                _home_short(file=file),
                _register_note(name=name, file=file, project=project),
            )
            for name, file in sorted(registered.items())
        ]
        log_table(logger=_LOGGER, header=('name', 'source', 'profile', 'note'), rows=rows)
        _LOGGER.info('use one: tslab study submit STUDY --machine <name>')
        return 0


def _project_machines(*, start: Path) -> dict[str, Path]:
    """The ``machines/<name>.toml`` files above ``start``; the nearest wins, as in --machine."""
    found: dict[str, Path] = {}
    for folder in (start, *start.parents):
        for file in sorted((folder / 'machines').glob('*.toml')):
            found.setdefault(file.stem, file)
    return found


def _home_short(*, file: Path) -> str:
    home = Path.home()
    if file.is_relative_to(home):
        return f'~/{file.relative_to(home).as_posix()}'
    return file.as_posix()


def _register_note(*, name: str, file: Path, project: dict[str, Path]) -> str:
    if name in project:
        return 'overridden by project'
    return '' if file.is_file() else 'missing'


@dataclass(frozen=True, kw_only=True)
class MachineRemove:
    """Remove a name from the register; the profile file stays.

    For example:
        tslab machine remove local

    Attributes:
        name: The registered name.
    """

    name: Annotated[str, tyro.conf.arg(metavar='NAME'), tyro.conf.Positional]

    def run(self) -> int:
        """Remove the name and return the exit code."""
        try:
            entries = read_register(environment=os.environ)
            if self.name not in entries:
                known = ', '.join(sorted(entries)) or 'none'
                message = f'machine {self.name!r} is not registered; registered names: {known}'
                raise MachineProfileError(message)  # noqa: TRY301 # why: one handler for all checks
            del entries[self.name]
            write_register(entries=entries, environment=os.environ)
        except MachineProfileError as error:
            raise UsageError(str(error)) from error
        _LOGGER.info('removed %s', self.name)
        return 0


MACHINE_COMMANDS = {'add': MachineAdd, 'list': MachineList, 'remove': MachineRemove}

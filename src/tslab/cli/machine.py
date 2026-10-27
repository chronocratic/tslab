"""The ``tslab machine`` subcommands: the register of machine names.

``--machine cluster`` finds ``machines/cluster.toml`` above the working
directory, and otherwise looks the name up in the register these commands edit.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Annotated

import tyro

from tslab.cli.usage import UsageError
from tslab.study.machine import (
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
    """Register a machine profile under a name, so ``--machine <name>`` finds it anywhere.

    Attributes:
        name: The name to register: letters, digits, ``_`` and ``-``.
        path: The machine profile file; stored as an absolute path.
    """

    name: Annotated[str, tyro.conf.Positional]
    path: Annotated[Path, tyro.conf.Positional]

    def run(self) -> int:
        """Validate the profile, store it and return the exit code."""
        try:
            validate_machine_name(name=self.name)
            file = self.path.expanduser().resolve()
            if not file.is_file():
                message = f'machine profile {file} does not exist; pass the path of a TOML file'
                raise MachineProfileError(message)  # noqa: TRY301 # why: one handler for all checks
            load_machine(path=file, lenient_env=True)
            write_register(entries={**read_register(), self.name: file})
        except MachineProfileError as error:
            raise UsageError(str(error)) from error
        _LOGGER.info('registered %s -> %s', self.name, file.as_posix())
        return 0


@dataclass(frozen=True, kw_only=True)
class MachineList:
    """Print the registered machine names and their profile files."""

    def run(self) -> int:
        """Print one line per name and return the exit code."""
        try:
            entries = read_register()
        except MachineProfileError as error:
            raise UsageError(str(error)) from error
        for name, file in sorted(entries.items()):
            note = '' if file.is_file() else '  (missing)'
            _LOGGER.info('%s  %s%s', name, file.as_posix(), note)
        return 0


@dataclass(frozen=True, kw_only=True)
class MachineRemove:
    """Remove a name from the register; the profile file stays.

    Attributes:
        name: The registered name.
    """

    name: Annotated[str, tyro.conf.Positional]

    def run(self) -> int:
        """Remove the name and return the exit code."""
        try:
            entries = read_register()
            if self.name not in entries:
                known = ', '.join(sorted(entries)) or 'none'
                message = f'machine {self.name!r} is not registered; registered names: {known}'
                raise MachineProfileError(message)  # noqa: TRY301 # why: one handler for all checks
            del entries[self.name]
            write_register(entries=entries)
        except MachineProfileError as error:
            raise UsageError(str(error)) from error
        _LOGGER.info('removed %s', self.name)
        return 0


MACHINE_COMMANDS = {'add': MachineAdd, 'list': MachineList, 'remove': MachineRemove}

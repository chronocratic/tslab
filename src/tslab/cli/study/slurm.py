"""The ``tslab study slurm`` command."""

from dataclasses import dataclass
import logging
from pathlib import Path
import sys
from typing import Annotated, cast

import tyro

from tslab.cli.common import resolve_request
from tslab.cli.study import _context_of
from tslab.core.meta import hostname
from tslab.jobs.slurm import run_slurm_command, slurm_command, SLURM_COMMANDS
from tslab.study.launchers.slurm.slurm_file import command_values
from tslab.study.paths import study_folder

__all__ = ['Slurm']

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class Slurm:
    """Run a read-only SLURM command for the jobs of a study; no name lists them.

    The commands are the ones of the slurm.sh file in the study folder, with
    the job ids of the study filled in and the full job name shown.

    For example:
        tslab study slurm example squeue-full-name

    Attributes:
        study: The short study name or full module:attribute reference. A short name expands
            against the studies package of tslab.toml.
        name: The command, <tool>-<variant>; leave out to list them.
        machine: A machine name or profile path. Default: from
            [machine_by_host] in tslab.toml.
    """

    study: Annotated[str, tyro.conf.arg(metavar='STUDY'), tyro.conf.Positional]
    name: Annotated[str | None, tyro.conf.arg(metavar='NAME'), tyro.conf.Positional] = None
    machine: Annotated[str | None, tyro.conf.arg(metavar='MACHINE', help_behavior_hint='')] = None

    def run(self) -> int:
        """Run the command and return its exit code."""
        if self.name is None:
            width = max(len(command.name) for command in SLURM_COMMANDS)
            for command in SLURM_COMMANDS:
                sys.stdout.write(f'{command.name:<{width}}  {command.description}\n')
            return 0
        try:
            command = slurm_command(name=self.name)
        except KeyError as error:
            _LOGGER.error('%s', error.args[0])  # noqa: TRY400 # why: a usage line, no traceback
            return 2
        request = resolve_request(
            study=self.study, machine=self.machine, start=Path.cwd(), host=hostname()
        )
        context = _context_of(
            study=cast('str', request.study),
            machine=cast('str', request.machine),
            grid=(),
            only=(),
            read_only=True,
        )
        folder = study_folder(machine=context.machine, study=context.study)
        values = command_values(study=context.study, study_folder=folder)
        if not values.job_ids:
            _LOGGER.error('no SLURM jobs recorded for this study; submit it first')
            return 1
        output = run_slurm_command(command=command, values=values)
        sys.stdout.write(output.stdout)
        sys.stderr.write(output.stderr)
        return output.returncode

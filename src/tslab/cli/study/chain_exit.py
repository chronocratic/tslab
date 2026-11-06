"""Internal CLI command that checks the run state of one chained SLURM task."""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, cast

import tyro

from tslab.cli.common import resolve_request
from tslab.cli.study.shared import _context_of
from tslab.core.meta import hostname
from tslab.study.commands.chain_exit import chain_exit

__all__ = ['ChainExit']


@dataclass(frozen=True, kw_only=True)
class ChainExit:
    """Return non-zero when one chained run did not finish successfully.

    For example:
        tslab study chain-exit example:study --machine machines/cluster.toml --job-file 3 --index 4

    Attributes:
        study: The full module:attribute reference of the study.
        machine: The machine profile used for the submission.
        job_file: The ``jobs/<n>`` file number of the submitted task.
        index: The array index of the task.
    """

    study: Annotated[str, tyro.conf.arg(metavar='STUDY'), tyro.conf.Positional]
    machine: Annotated[str, tyro.conf.arg(metavar='MACHINE')]
    job_file: int
    index: int

    def run(self) -> int:
        """Read the task run status and return its chained dependency exit code.

        Returns:
            The exit code of the task state check.
        """
        request = resolve_request(
            study=self.study, machine=self.machine, start=Path.cwd(), host=hostname()
        )
        context = _context_of(
            study=cast('str', request.study),  # why: this command requires a study reference
            machine=cast('str', request.machine),  # why: machine is a required CLI argument
            grid=(),
            only=(),
            read_only=True,
        )
        return chain_exit(context=context, job_file=self.job_file, index=self.index)

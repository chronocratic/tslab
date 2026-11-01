"""The ``tslab study failures`` and ``usage`` commands."""

from dataclasses import dataclass
import logging
from typing import Annotated

import tyro

from tslab.cli.output import print_json
from tslab.cli.study import _context_of, _states_of
from tslab.jobs.slurm import SlurmError
from tslab.study.failures import collect_failures, log_failures
from tslab.study.launchers import backend_for
from tslab.study.usage import log_usage, usage_document

__all__ = ['Failures', 'Usage']

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class Failures:
    """Show error details and recent logs for failed or interrupted runs.

    For example:
        tslab study failures my_paper.studies:example --machine local

    Attributes:
        study: The study reference, module:attribute.
        machine: A machine name or profile path.
        grid: Keep only this grid.
        only: Keep only runs that match axis=value.
        states: Show only runs in these states; failed and interrupted by default.
        lines: Number of trailing lines to show from each log.
        json: Print one study-failures JSON document.
    """

    study: Annotated[str, tyro.conf.arg(metavar='STUDY'), tyro.conf.Positional]
    machine: Annotated[str, tyro.conf.arg(metavar='MACHINE')]
    grid: Annotated[tuple[str, ...], tyro.conf.arg(metavar='GRID'), tyro.conf.UseAppendAction] = ()
    only: Annotated[
        tuple[str, ...], tyro.conf.arg(metavar='AXIS=VALUES'), tyro.conf.UseAppendAction
    ] = ()
    states: Annotated[
        tuple[str, ...], tyro.conf.arg(metavar='STATE'), tyro.conf.UseAppendAction
    ] = ()
    lines: int = 30
    json: bool = False

    def run(self) -> int:
        """Show failure details and return the process exit code."""
        context = _context_of(
            study=self.study, machine=self.machine, grid=self.grid, only=self.only, read_only=True
        )
        failures = collect_failures(
            context=context, states=_states_of(values=self.states), lines=self.lines
        )
        if self.json:
            print_json(document={'schema': 'tslab.study-failures/1', 'runs': failures})
        elif failures:
            log_failures(failures=failures)
        else:
            _LOGGER.info('no failed or interrupted runs')
        return 0


@dataclass(frozen=True, kw_only=True)
class Usage:
    """Show what each run used, and the sizes to suggest for the next jobs.

    For example:
        tslab study usage my_paper.studies:example --machine cluster

    Attributes:
        study: The study reference, module:attribute.
        machine: A machine name or profile path.
        grid: Keep only this grid.
        only: Keep only runs that match axis=value.
        json: Print one study-usage JSON document.
    """

    study: Annotated[str, tyro.conf.arg(metavar='STUDY'), tyro.conf.Positional]
    machine: Annotated[str, tyro.conf.arg(metavar='MACHINE')]
    grid: Annotated[tuple[str, ...], tyro.conf.arg(metavar='GRID'), tyro.conf.UseAppendAction] = ()
    only: Annotated[
        tuple[str, ...], tyro.conf.arg(metavar='AXIS=VALUES'), tyro.conf.UseAppendAction
    ] = ()
    json: bool = False

    def run(self) -> int:
        """Show the usage of the selected runs and return the process exit code."""
        context = _context_of(
            study=self.study, machine=self.machine, grid=self.grid, only=self.only, read_only=True
        )
        try:
            document = usage_document(context=context)
        except SlurmError as error:
            _LOGGER.error('%s', error)  # noqa: TRY400 # why: scheduler errors are shown, not raised
            return 1
        if self.json:
            print_json(document=document)
        else:
            log_usage(
                document=document,
                suggests_memory=backend_for(machine=context.machine).suggests_memory,
            )
        return 0

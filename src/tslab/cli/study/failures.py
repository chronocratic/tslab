"""The ``tslab study failures`` and ``usage`` commands."""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Annotated, cast

import tyro

from tslab.cli.common import print_json, resolve_request
from tslab.cli.study import _context_of, _states_of
from tslab.core.meta import hostname
from tslab.study.launchers import backend_for, LaunchError
from tslab.study.reporting.failures import collect_failures, log_failures
from tslab.study.reporting.usage import log_usage, usage_document

__all__ = ['Failures', 'Usage']

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class Failures:
    """Show error details and recent logs for failed or interrupted runs.

    For example:
        tslab study failures my_paper.studies:example --machine local

    Attributes:
        study: The study reference, module:attribute. A short name is
            expanded against the studies module of tslab.toml.
        machine: A machine name or profile path. Default: from
            [machine_by_host] in tslab.toml.
        grid: Keep only this grid.
        only: Keep only runs that match axis=value.
        states: Show only runs in these states; failed and interrupted by default.
        lines: Number of trailing lines to show from each log.
        json: Print one study-failures JSON document.
    """

    study: Annotated[str, tyro.conf.arg(metavar='STUDY'), tyro.conf.Positional]
    machine: Annotated[str | None, tyro.conf.arg(metavar='MACHINE', help_behavior_hint='')] = None
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
        request = resolve_request(
            study=self.study, machine=self.machine, start=Path.cwd(), host=hostname()
        )
        study = cast('str', request.study)
        machine = cast('str', request.machine)
        context = _context_of(
            study=study, machine=machine, grid=self.grid, only=self.only, read_only=True
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
        study: The study reference, module:attribute. A short name is
            expanded against the studies module of tslab.toml.
        machine: A machine name or profile path. Default: from
            [machine_by_host] in tslab.toml.
        grid: Keep only this grid.
        only: Keep only runs that match axis=value.
        json: Print one study-usage JSON document.
    """

    study: Annotated[str, tyro.conf.arg(metavar='STUDY'), tyro.conf.Positional]
    machine: Annotated[str | None, tyro.conf.arg(metavar='MACHINE', help_behavior_hint='')] = None
    grid: Annotated[tuple[str, ...], tyro.conf.arg(metavar='GRID'), tyro.conf.UseAppendAction] = ()
    only: Annotated[
        tuple[str, ...], tyro.conf.arg(metavar='AXIS=VALUES'), tyro.conf.UseAppendAction
    ] = ()
    json: bool = False

    def run(self) -> int:
        """Show the usage of the selected runs and return the process exit code."""
        request = resolve_request(
            study=self.study, machine=self.machine, start=Path.cwd(), host=hostname()
        )
        study = cast('str', request.study)
        machine = cast('str', request.machine)
        context = _context_of(
            study=study, machine=machine, grid=self.grid, only=self.only, read_only=True
        )
        try:
            document = usage_document(context=context)
        except LaunchError as error:
            _LOGGER.error('%s', error)  # noqa: TRY400 # why: scheduler errors are shown, not raised
            return 1
        if self.json:
            print_json(document=document)
        else:
            log_usage(
                document=document,
                has_time_limit=backend_for(machine=context.machine).has_time_limit,
            )
        return 0

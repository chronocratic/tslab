"""The ``tslab study submit`` command."""

from collections.abc import Sequence
from dataclasses import dataclass
import logging
import sys
from typing import Annotated

import tyro

from tslab.cli.common import print_json
from tslab.cli.study import _context_of, _detach, _selection_args, _selection_options, _warn_no_runs
from tslab.study.commands import submit
from tslab.study.launchers import LaunchReport
from tslab.study.plan import JobGroup
from tslab.study.reporting.documents import launch_document

__all__ = ['Submit']

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class Submit:
    """Submit the runs of a study to a machine, or dry-run the plan.

    For example:
        tslab study submit my_paper.studies:example --machine local --dry-run
        tslab study submit my_paper.studies:example --machine local -- --trainer.max-epochs 5

    Attributes:
        study: The study reference, module:attribute.
        machine: A machine name (machines/<name>.toml above the working folder, or
            registered with tslab machine add) or a profile path.
        grid: Keep only this grid.
        only: Keep only the runs that match axis=value; comma separated for
            several values of one axis.
        include_done: Submit the done runs as well.
        dry_run: Write the job files and print the plan, submit nothing.
        detach: Local machines only: start the launch in the background and return once
            it runs; follow it with tslab study wait.
        extra_args: The arguments after --, appended to every run.
        json: Print one study-launch JSON document.
    """

    study: Annotated[str, tyro.conf.arg(metavar='STUDY'), tyro.conf.Positional]
    machine: Annotated[str, tyro.conf.arg(metavar='MACHINE')]
    grid: Annotated[tuple[str, ...], tyro.conf.arg(metavar='GRID'), tyro.conf.UseAppendAction] = ()
    only: Annotated[
        tuple[str, ...], tyro.conf.arg(metavar='AXIS=VALUES'), tyro.conf.UseAppendAction
    ] = ()
    include_done: bool = False
    dry_run: bool = False
    detach: bool = False
    extra_args: tyro.conf.Suppress[tuple[str, ...]] = ()
    json: bool = False

    def run(self) -> int:
        """Submit and return the exit code."""
        context = _context_of(
            study=self.study, machine=self.machine, grid=self.grid, only=self.only
        )
        if self.detach:
            flags = [
                flag
                for flag, on in (
                    ('--include-done', self.include_done),
                    ('--dry-run', self.dry_run),
                    ('--json', self.json),
                )
                if on
            ]
            return _detach(
                context=context,
                command='submit',
                selection_options=_selection_options(grid=self.grid, only=self.only),
                options=[],
                flags=flags,
                extra_args=self.extra_args,
                request=(self.study, self.machine),
            )
        _warn_no_runs(context=context)

        launches: list[tuple[LaunchReport, Sequence[JobGroup]]] = []

        def remember_launch(*, report: LaunchReport, groups: Sequence[JobGroup]) -> None:
            launches.append((report, groups))

        code = submit(
            context=context,
            include_done=self.include_done,
            dry_run=self.dry_run,
            extra_args=self.extra_args,
            on_launch=remember_launch if self.json else None,
            task_stream=sys.stderr if self.json else None,
        )
        if code == 0:
            if self.dry_run:
                _LOGGER.info('dry run: nothing submitted; run again without --dry-run to submit')
            else:
                selection = _selection_args(
                    study=self.study, machine=self.machine, grid=self.grid, only=self.only
                )
                _LOGGER.info('follow progress: tslab study status %s', selection)
        if self.json and (code == 0 or launches):
            report, groups = launches[0] if launches else (None, ())
            print_json(
                document=launch_document(
                    report=report, groups=groups, dry_run=self.dry_run, cancelled=(), exit_code=code
                )
            )
        return code

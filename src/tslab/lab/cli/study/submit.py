"""The ``tslab study submit`` command."""

from collections.abc import Sequence
from dataclasses import dataclass
import logging
from pathlib import Path
import sys
from typing import Annotated, cast

import tyro

from tslab.lab.cli.common import print_json, resolve_request, UsageError
from tslab.lab.cli.study.shared import (
    _context_of,
    _detach,
    _log_next_step,
    _selection_args,
    _selection_options,
    _warn_no_runs,
    take_requested_code_snapshot,
)
from tslab.lab.core.meta import hostname
from tslab.lab.study.commands import submit
from tslab.lab.study.launchers.common.types import LaunchReport
from tslab.lab.study.plan import JobGroup
from tslab.lab.study.reporting.documents import launch_document

__all__ = ['Submit']

_LOGGER = logging.getLogger(__name__)


def _check_snapshot_flags(
    *, code_snapshot: bool, no_code_snapshot: bool, include_untracked: bool, wanted: bool
) -> None:
    if code_snapshot and no_code_snapshot:
        message = '--code-snapshot and --no-code-snapshot are mutually exclusive'
        raise UsageError(message)
    if include_untracked and not wanted:
        message = (
            '--code-snapshot-include-untracked needs a code snapshot: pass --code-snapshot, '
            'or set code_snapshot = true in tslab.toml'
        )
        raise UsageError(message)


@dataclass(frozen=True, kw_only=True)
class Submit:
    """Submit the runs of a study to a machine, or dry-run the plan.

    For example:
        tslab study submit example --machine local --dry-run
        tslab study submit example --machine local -- --trainer.max-epochs 5

    Attributes:
        study: The short study name or full module:attribute reference. A short name expands
            against the studies package of tslab.toml.
        machine: A machine name (machines/<name>.toml above the working folder, or
            registered with tslab machine add) or a profile path. Default: from
            [machine_by_host] in tslab.toml.
        grid: Keep only this grid.
        only: Keep only the runs that match axis=value; comma separated for
            several values of one axis.
        include_done: Submit the done runs as well.
        dry_run: Write the job files and print the plan, submit nothing.
        detach: Local machines only: start the launch in the background and return once
            it runs; follow it with tslab study wait.
        code_snapshot: Copy src/ into a content-addressed folder and run the jobs
            from that copy; code_snapshot = true in tslab.toml does the same.
        no_code_snapshot: Run the live code, even when tslab.toml turns the code
            snapshot on.
        code_snapshot_include_untracked: With a code snapshot, copy untracked
            files under src/ instead of stopping.
        ignore_after: Submit without dependency targets or the local dependency filter.
        extra_args: The arguments after --, appended to every run.
        json: Print one study-launch JSON document.
    """

    study: Annotated[str, tyro.conf.arg(metavar='STUDY'), tyro.conf.Positional]
    machine: Annotated[str | None, tyro.conf.arg(metavar='MACHINE', help_behavior_hint='')] = None
    grid: Annotated[tuple[str, ...], tyro.conf.arg(metavar='GRID'), tyro.conf.UseAppendAction] = ()
    only: Annotated[
        tuple[str, ...], tyro.conf.arg(metavar='AXIS=VALUES'), tyro.conf.UseAppendAction
    ] = ()
    include_done: bool = False
    dry_run: bool = False
    detach: bool = False
    code_snapshot: bool = False
    no_code_snapshot: bool = False
    code_snapshot_include_untracked: bool = False
    ignore_after: bool = False
    extra_args: tyro.conf.Suppress[tuple[str, ...]] = ()
    json: bool = False

    def run(self) -> int:
        """Submit and return the exit code."""
        request = resolve_request(
            study=self.study, machine=self.machine, start=Path.cwd(), host=hostname()
        )
        study = cast('str', request.study)
        machine = cast('str', request.machine)
        context = _context_of(study=study, machine=machine, grid=self.grid, only=self.only)
        wanted = self.code_snapshot or (
            request.settings.code_snapshot and not self.no_code_snapshot
        )
        _check_snapshot_flags(
            code_snapshot=self.code_snapshot,
            no_code_snapshot=self.no_code_snapshot,
            include_untracked=self.code_snapshot_include_untracked,
            wanted=wanted,
        )
        if self.detach:
            flags = [
                flag
                for flag, on in (
                    ('--include-done', self.include_done),
                    ('--dry-run', self.dry_run),
                    ('--json', self.json),
                    ('--code-snapshot', self.code_snapshot),
                    ('--no-code-snapshot', self.no_code_snapshot),
                    ('--code-snapshot-include-untracked', self.code_snapshot_include_untracked),
                    ('--ignore-after', self.ignore_after),
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
                request=request,
                wanted=wanted,
                include_untracked=self.code_snapshot_include_untracked,
            )
        # why: a detached launch repeats this command in the background, and that child
        # takes the snapshot, so taking it here too would copy src/ twice
        code_snapshot = take_requested_code_snapshot(
            request=request,
            context=context,
            wanted=wanted,
            include_untracked=self.code_snapshot_include_untracked,
            dry_run=self.dry_run,
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
            code_snapshot=code_snapshot,
            ignore_after=self.ignore_after,
            on_launch=remember_launch if self.json else None,
            task_stream=sys.stderr if self.json else None,
        )
        if code == 0:
            _log_next_step(
                dry_run=self.dry_run,
                selection=_selection_args(
                    study=study, machine=machine, grid=self.grid, only=self.only
                ),
            )
        if self.json and (code == 0 or launches):
            report, groups = launches[0] if launches else (None, ())
            print_json(
                document=launch_document(
                    report=report, groups=groups, dry_run=self.dry_run, cancelled=(), exit_code=code
                )
            )
        return code

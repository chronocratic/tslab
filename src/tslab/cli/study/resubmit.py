"""The ``tslab study resubmit`` and ``run`` commands."""

from collections.abc import Sequence
from dataclasses import dataclass
import logging
from pathlib import Path
import sys
from typing import Annotated, cast

import tyro

from tslab.cli.common import print_json, resolve_request, UsageError
from tslab.cli.study import (
    _context_of,
    _detach,
    _log_next_step,
    _selection_args,
    _selection_options,
    _states_of,
    _warn_no_runs,
    take_requested_code_snapshot,
)
from tslab.core.meta import hostname
from tslab.study.code_snapshot import CodeSnapshotError
from tslab.study.commands import resubmit, run_one
from tslab.study.launchers.common.types import LaunchReport
from tslab.study.launchers.tables import backend_for
from tslab.study.plan import JobGroup
from tslab.study.reporting.documents import launch_document

__all__ = ['Resubmit', 'RunOne']

_LOGGER = logging.getLogger(__name__)


def _check_resubmit_snapshot_flags(*, new: bool, drop: bool, include_untracked: bool) -> None:
    if new and drop:
        message = '--new-code-snapshot and --no-code-snapshot are mutually exclusive'
        raise UsageError(message)
    if include_untracked and not new:
        message = '--code-snapshot-include-untracked needs --new-code-snapshot'
        raise UsageError(message)


@dataclass(frozen=True, kw_only=True)
class RunOne:
    """Run one selected run in the foreground of this terminal.

    The command of the run is built the same way as for a submission, but
    nothing is submitted or recorded: SLURM is never called and the study
    folder is untouched. Inside a ``salloc`` allocation, ``job_time_share``
    still pauses the training; pass ``-- --trainer.job-time-share 1`` to
    train for the whole allocation.

    For example:
        tslab study run my_paper.studies:example --machine cluster --only model=small,seed=1

    Attributes:
        study: The study reference, module:attribute. A short name is
            expanded against the studies module of tslab.toml.
        machine: A machine name or profile path. Default: from
            [machine_by_host] in tslab.toml.
        grid: Keep only this grid.
        only: Keep only runs that match axis=value.
        extra_args: The arguments after --; taken from the last
            submission when omitted.
    """

    study: Annotated[str, tyro.conf.arg(metavar='STUDY'), tyro.conf.Positional]
    machine: Annotated[str | None, tyro.conf.arg(metavar='MACHINE', help_behavior_hint='')] = None
    grid: Annotated[tuple[str, ...], tyro.conf.arg(metavar='GRID'), tyro.conf.UseAppendAction] = ()
    only: Annotated[
        tuple[str, ...], tyro.conf.arg(metavar='AXIS=VALUES'), tyro.conf.UseAppendAction
    ] = ()
    extra_args: tyro.conf.Suppress[tuple[str, ...]] = ()

    def run(self) -> int:
        """Run the one selected run and return its exit code."""
        request = resolve_request(
            study=self.study, machine=self.machine, start=Path.cwd(), host=hostname()
        )
        study = cast('str', request.study)
        machine = cast('str', request.machine)
        context = _context_of(study=study, machine=machine, grid=self.grid, only=self.only)
        try:
            return run_one(context=context, extra_args=self.extra_args or None)
        except ValueError as error:
            raise UsageError(str(error)) from error


@dataclass(frozen=True, kw_only=True)
class Resubmit:
    """Submit the interrupted, missing or failed runs of a study again.

    For example:
        tslab study resubmit my_paper.studies:example --machine local
        tslab study resubmit my_paper.studies:example --machine local --dry-run

    Attributes:
        study: The study reference, module:attribute. A short name is
            expanded against the studies module of tslab.toml.
        machine: A machine name (machines/<name>.toml above the working folder, or
            registered with tslab machine add) or a profile path. Default: from
            [machine_by_host] in tslab.toml.
        grid: Keep only this grid.
        only: Keep only the runs that match axis=value; comma separated for
            several values of one axis.
        states: The states to resubmit; comma separated. Interrupted and
            missing when omitted.
        cancel_live: Cancel the queued and running runs first and
            resubmit them a little later.
        dry_run: Write the job files and print the plan, submit nothing.
        auto: Retry only the interrupted runs that made progress, up to
            followup.max_attempts each; ignores --states and --cancel-live.
            The follow-up jobs pass this.
        detach: Local machines only: start the launch in the background and return once
            it runs; follow it with tslab study wait.
        new_code_snapshot: Copy src/ into a fresh content-addressed folder and
            run the resubmitted jobs from it, replacing the recorded snapshot.
        no_code_snapshot: Run the live code, even when the runs recorded a code
            snapshot.
        code_snapshot_include_untracked: With --new-code-snapshot, copy
            untracked files under src/ instead of stopping.
        extra_args: The arguments after --; taken from the last
            submission when omitted.
        json: Print one study-launch JSON document.
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
    cancel_live: bool = False
    dry_run: bool = False
    auto: bool = False
    detach: bool = False
    new_code_snapshot: bool = False
    no_code_snapshot: bool = False
    code_snapshot_include_untracked: bool = False
    extra_args: tyro.conf.Suppress[tuple[str, ...]] = ()
    json: bool = False

    def run(self) -> int:
        """Resubmit and return the exit code."""
        request = resolve_request(
            study=self.study, machine=self.machine, start=Path.cwd(), host=hostname()
        )
        study = cast('str', request.study)
        machine = cast('str', request.machine)
        states = _states_of(values=self.states)
        context = _context_of(study=study, machine=machine, grid=self.grid, only=self.only)
        # why: the code snapshot of tslab.toml does not apply to a resubmit; the runs
        # keep the snapshot of their latest job unless the flags say otherwise
        _check_resubmit_snapshot_flags(
            new=self.new_code_snapshot,
            drop=self.no_code_snapshot,
            include_untracked=self.code_snapshot_include_untracked,
        )
        code_snapshot = take_requested_code_snapshot(
            request=request,
            context=context,
            wanted=self.new_code_snapshot,
            include_untracked=self.code_snapshot_include_untracked,
            dry_run=self.dry_run,
        )
        if self.detach:
            flags = [
                flag
                for flag, on in (
                    ('--cancel-live', self.cancel_live),
                    ('--dry-run', self.dry_run),
                    ('--auto', self.auto),
                    ('--json', self.json),
                    ('--new-code-snapshot', self.new_code_snapshot),
                    ('--no-code-snapshot', self.no_code_snapshot),
                    ('--code-snapshot-include-untracked', self.code_snapshot_include_untracked),
                )
                if on
            ]
            options: list[str] = []
            for state in self.states:
                options.extend(('--states', state))
            return _detach(
                context=context,
                command='resubmit',
                selection_options=_selection_options(grid=self.grid, only=self.only),
                options=options,
                flags=flags,
                extra_args=self.extra_args,
                request=(study, machine),
            )
        _warn_no_runs(context=context)
        launches: list[tuple[LaunchReport, Sequence[JobGroup]]] = []

        def remember_launch(*, report: LaunchReport, groups: Sequence[JobGroup]) -> None:
            launches.append((report, groups))

        backend = backend_for(machine=context.machine)
        cancelled = (
            backend.cancellable_job_ids(reports=context.reports)
            if self.cancel_live and not self.auto
            else []
        )
        try:
            code = resubmit(
                context=context,
                states=states,
                cancel_live=self.cancel_live,
                dry_run=self.dry_run,
                extra_args=self.extra_args or None,
                auto=self.auto,
                new_code_snapshot=code_snapshot,
                drop_code_snapshot=self.no_code_snapshot,
                on_launch=remember_launch if self.json else None,
                task_stream=sys.stderr if self.json else None,
            )
        except CodeSnapshotError as error:
            raise UsageError(str(error)) from error
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
                    report=report,
                    groups=groups,
                    dry_run=self.dry_run,
                    cancelled=cancelled,
                    exit_code=code,
                )
            )
        return code

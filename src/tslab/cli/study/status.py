"""The ``tslab study status``, ``show`` and ``wait`` commands."""

from dataclasses import dataclass
from typing import Annotated

import tyro

from tslab.cli.common import print_json, UsageError, watch
from tslab.cli.study import _context_of, _states_of, _status_hints, _usage_errors
from tslab.study.commands import status
from tslab.study.context import Context, expected_starts
from tslab.study.documents import show_document, status_document
from tslab.study.loading import load_study
from tslab.study.outcome import EXIT_TIMEOUT, study_exit_code
from tslab.study.report import log_show
from tslab.study.wait import wait_for_study

__all__ = ['Show', 'Status', 'Wait']

_MIN_WAIT_INTERVAL_SECONDS = 10


@dataclass(frozen=True, kw_only=True)
class Status:
    """Show the state of every run of a study, and the runs not done.

    For example:
        tslab study status my_paper.studies:example --machine local
        tslab study status my_paper.studies:example --machine local --watch 60

    Attributes:
        study: The study reference, module:attribute.
        machine: A machine name (machines/<name>.toml above the working folder, or
            registered with tslab machine add) or a profile path.
        grid: Keep only this grid.
        only: Keep only the runs that match axis=value; comma separated for
            several values of one axis.
        states: Show only the runs in these states; comma separated.
        watch: Refresh every this many seconds until Ctrl-C.
        json: Print one study-status JSON document.
        exit_code: Return the code for the run states.
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
    watch: Annotated[float | None, tyro.conf.arg(metavar='SECONDS', help_behavior_hint='')] = None
    json: bool = False
    exit_code: bool = False

    def run(self) -> int:
        """Print the table and return the exit code."""
        if self.watch is not None and self.json:
            message = '--watch does not combine with --json; use tslab study wait'
            raise UsageError(message)
        states = _states_of(values=self.states)

        def show() -> int:
            context = _context_of(
                study=self.study,
                machine=self.machine,
                grid=self.grid,
                only=self.only,
                read_only=True,
            )
            if self.exit_code and not context.reports:
                message = 'no runs match the selection; drop --grid/--only or check values'
                raise UsageError(message)
            starts = expected_starts(context=context)
            status(context=context, states=states, starts=starts)
            _status_hints(
                context=context,
                study=self.study,
                machine=self.machine,
                grid=self.grid,
                only=self.only,
            )
            code = study_exit_code(context=context)
            if self.json:
                print_json(document=status_document(context=context, exit_code=code, starts=starts))
            return code if self.exit_code else 0

        return watch(seconds=self.watch, show=show)


@dataclass(frozen=True, kw_only=True)
class Show:
    """Show the grids, axes, and run counts of a study.

    For example:
        tslab study show my_paper.studies:example

    Attributes:
        study: The study reference, module:attribute.
        json: Print one study-show JSON document.
    """

    study: Annotated[str, tyro.conf.arg(metavar='STUDY'), tyro.conf.Positional]
    json: bool = False

    def run(self) -> int:
        """Show a study without loading a machine profile."""
        with _usage_errors(study=self.study):
            study = load_study(reference=self.study)
        if self.json:
            print_json(document=show_document(study=study))
        else:
            log_show(study=study)
        return 0


@dataclass(frozen=True, kw_only=True)
class Wait:
    """Wait for selected runs to finish or need attention.

    For example:
        tslab study wait my_paper.studies:example --machine local

    Attributes:
        study: The study reference, module:attribute.
        machine: A machine name or profile path.
        grid: Keep only this grid.
        only: Keep only runs that match axis=value.
        interval: Longest gap between checks in seconds (at least 10); checks start
            30 seconds apart and slow down while nothing changes.
        timeout: Stop after this many seconds.
        keep_going: Wait until no run is live, even after a failure.
        json: Print one study-wait JSON document.
    """

    study: Annotated[str, tyro.conf.arg(metavar='STUDY'), tyro.conf.Positional]
    machine: Annotated[str, tyro.conf.arg(metavar='MACHINE')]
    grid: Annotated[tuple[str, ...], tyro.conf.arg(metavar='GRID'), tyro.conf.UseAppendAction] = ()
    only: Annotated[
        tuple[str, ...], tyro.conf.arg(metavar='AXIS=VALUES'), tyro.conf.UseAppendAction
    ] = ()
    interval: Annotated[float, tyro.conf.arg(metavar='SECONDS')] = 300.0
    timeout: Annotated[float | None, tyro.conf.arg(metavar='SECONDS')] = None
    keep_going: bool = False
    json: bool = False

    def run(self) -> int:
        """Wait for the study and return its final state code."""
        if self.interval < _MIN_WAIT_INTERVAL_SECONDS:
            message = '--interval takes at least 10 seconds'
            raise UsageError(message)

        ended_states: dict[str, str] = {}

        def load() -> Context:
            context = _context_of(
                study=self.study,
                machine=self.machine,
                grid=self.grid,
                only=self.only,
                read_only=True,
                ended_states=ended_states,
            )
            if not context.reports:
                message = 'no runs match the selection; drop --grid/--only or check values'
                raise UsageError(message)
            return context

        code, context, waited = wait_for_study(
            load=load, interval=self.interval, timeout=self.timeout, keep_going=self.keep_going
        )
        if self.json:
            document = status_document(context=context, exit_code=code)
            document['schema'] = 'tslab.study-wait/1'
            document['waited_seconds'] = waited
            document['timed_out'] = code == EXIT_TIMEOUT
            print_json(document=document)
        else:
            status(context=context)
            _status_hints(
                context=context,
                study=self.study,
                machine=self.machine,
                grid=self.grid,
                only=self.only,
            )
        return code

"""The ``tslab study`` subcommands: thin tyro dataclasses over ``tslab.study.commands``.

The argument dataclasses here hold only the flags of one command; the
behavior lives in ``tslab.study.commands``, which the ``run`` methods call.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
import shlex
from typing import Annotated

import tyro

from tslab.cli.console import watch
from tslab.cli.usage import UsageError
from tslab.study.commands import parse_states, resubmit, results, status, submit
from tslab.study.context import Context, load_context, Selection
from tslab.study.machine import resolve_machine
from tslab.study.state import StudyState

__all__ = ['STUDY_COMMANDS', 'Resubmit', 'Results', 'Status', 'Submit']

_LOGGER = logging.getLogger(__name__)


def _context_of(
    *,
    study: str,
    machine: str,
    grid: tuple[str, ...],
    only: tuple[str, ...],
    read_only: bool = False,
) -> Context:
    """The context of a study reference; a bad reference or profile is a usage error."""
    try:
        location = resolve_machine(machine=machine, start=Path.cwd())
        return load_context(
            reference=study,
            machine_path=location.path,
            selection=Selection(grids=grid, only=only),
            project_root=location.project_root,
            read_only=read_only,
        )
    except ModuleNotFoundError as error:
        module_name = study.partition(':')[0]
        message = str(error)
        # why: only a missing study module is a usage error worth a hint; a missing
        # dependency inside it keeps its own message
        if error.name is not None and f'{module_name}.'.startswith(f'{error.name}.'):
            message = (
                f'cannot import the study module {module_name!r} ({error}); run tslab from '
                'the project folder, or check STUDY (module:attribute)'
            )
        raise UsageError(message) from error
    except (ValueError, TypeError, FileNotFoundError) as error:
        raise UsageError(str(error)) from error


def _states_of(*, values: tuple[str, ...]) -> tuple[StudyState, ...]:
    try:
        return parse_states(values=values)
    except ValueError as error:
        raise UsageError(str(error)) from error


def _selection_args(
    *, study: str, machine: str, grid: tuple[str, ...], only: tuple[str, ...]
) -> str:
    arguments = [study, '--machine', machine]
    for grid_name in grid:
        arguments.extend(('--grid', grid_name))
    for selection in only:
        arguments.extend(('--only', selection))
    return shlex.join(arguments)


def _warn_no_runs(*, context: Context) -> None:
    if not context.reports:
        _LOGGER.warning('no runs match the selection; drop --grid/--only or check their values')


def _status_hints(
    *, context: Context, study: str, machine: str, grid: tuple[str, ...], only: tuple[str, ...]
) -> None:
    if not context.reports:
        _warn_no_runs(context=context)
        return
    selection = _selection_args(study=study, machine=machine, grid=grid, only=only)
    states = {report.state for report in context.reports}
    if states == {StudyState.MISSING}:
        _LOGGER.info('nothing ran yet; check the plan: tslab study submit %s --dry-run', selection)
        return
    if StudyState.FAILED in states:
        _LOGGER.info(
            'fix the failed runs, then: tslab study resubmit %s --states failed', selection
        )
    if StudyState.INTERRUPTED in states or StudyState.MISSING in states:
        _LOGGER.info('retry interrupted and missing runs: tslab study resubmit %s', selection)
    if StudyState.QUEUED in states or StudyState.RUNNING in states:
        _LOGGER.info('follow progress: tslab study status %s --watch 60', selection)
    if all(report.state is StudyState.DONE for report in context.reports):
        _LOGGER.info(
            'all %d runs done; collect metrics: tslab study results %s --out results.csv',
            len(context.reports),
            selection,
        )


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
        extra_args: The arguments after --, appended to every run.
    """

    study: Annotated[str, tyro.conf.arg(metavar='STUDY'), tyro.conf.Positional]
    machine: Annotated[str, tyro.conf.arg(metavar='MACHINE')]
    grid: Annotated[tuple[str, ...], tyro.conf.arg(metavar='GRID'), tyro.conf.UseAppendAction] = ()
    only: Annotated[
        tuple[str, ...], tyro.conf.arg(metavar='AXIS=VALUES'), tyro.conf.UseAppendAction
    ] = ()
    include_done: bool = False
    dry_run: bool = False
    extra_args: tyro.conf.Suppress[tuple[str, ...]] = ()

    def run(self) -> int:
        """Submit and return the exit code."""
        context = _context_of(
            study=self.study, machine=self.machine, grid=self.grid, only=self.only
        )
        _warn_no_runs(context=context)
        code = submit(
            context=context,
            include_done=self.include_done,
            dry_run=self.dry_run,
            extra_args=self.extra_args,
        )
        if code == 0:
            if self.dry_run:
                _LOGGER.info('dry run: nothing submitted; run again without --dry-run to submit')
            else:
                selection = _selection_args(
                    study=self.study, machine=self.machine, grid=self.grid, only=self.only
                )
                _LOGGER.info('follow progress: tslab study status %s', selection)
        return code


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

    def run(self) -> int:
        """Print the table and return the exit code."""
        states = _states_of(values=self.states)

        def show() -> int:
            context = _context_of(
                study=self.study,
                machine=self.machine,
                grid=self.grid,
                only=self.only,
                read_only=True,
            )
            code = status(context=context, states=states)
            _status_hints(
                context=context,
                study=self.study,
                machine=self.machine,
                grid=self.grid,
                only=self.only,
            )
            return code

        return watch(seconds=self.watch, show=show)


@dataclass(frozen=True, kw_only=True)
class Resubmit:
    """Submit the interrupted, missing or failed runs of a study again.

    For example:
        tslab study resubmit my_paper.studies:example --machine local
        tslab study resubmit my_paper.studies:example --machine local --dry-run

    Attributes:
        study: The study reference, module:attribute.
        machine: A machine name (machines/<name>.toml above the working folder, or
            registered with tslab machine add) or a profile path.
        grid: Keep only this grid.
        only: Keep only the runs that match axis=value; comma separated for
            several values of one axis.
        states: The states to resubmit; comma separated. Interrupted and
            missing when omitted.
        cancel_live: Cancel the queued and running runs first and
            resubmit them a little later.
        dry_run: Write the job files and print the plan, submit nothing.
        auto: Set by the follow-up jobs of TSLab; not for use by hand.
        extra_args: The arguments after --; taken from the last
            submission when omitted.
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
    cancel_live: bool = False
    dry_run: bool = False
    auto: bool = False
    extra_args: tyro.conf.Suppress[tuple[str, ...]] = ()

    def run(self) -> int:
        """Resubmit and return the exit code."""
        states = _states_of(values=self.states)
        context = _context_of(
            study=self.study, machine=self.machine, grid=self.grid, only=self.only
        )
        _warn_no_runs(context=context)
        code = resubmit(
            context=context,
            states=states,
            cancel_live=self.cancel_live,
            dry_run=self.dry_run,
            extra_args=self.extra_args or None,
            auto=self.auto,
        )
        if code == 0:
            if self.dry_run:
                _LOGGER.info('dry run: nothing submitted; run again without --dry-run to submit')
            else:
                selection = _selection_args(
                    study=self.study, machine=self.machine, grid=self.grid, only=self.only
                )
                _LOGGER.info('follow progress: tslab study status %s', selection)
        return code


@dataclass(frozen=True, kw_only=True)
class Results:
    """Collect the metrics of every run of a study into one CSV file.

    For example:
        tslab study results my_paper.studies:example --machine local --out results.csv

    Attributes:
        study: The study reference, module:attribute.
        machine: A machine name (machines/<name>.toml above the working folder, or
            registered with tslab machine add) or a profile path.
        grid: Keep only this grid.
        only: Keep only the runs that match axis=value; comma separated for
            several values of one axis.
        out: The CSV file to write.
    """

    study: Annotated[str, tyro.conf.arg(metavar='STUDY'), tyro.conf.Positional]
    machine: Annotated[str, tyro.conf.arg(metavar='MACHINE')]
    grid: Annotated[tuple[str, ...], tyro.conf.arg(metavar='GRID'), tyro.conf.UseAppendAction] = ()
    only: Annotated[
        tuple[str, ...], tyro.conf.arg(metavar='AXIS=VALUES'), tyro.conf.UseAppendAction
    ] = ()
    out: Annotated[Path, tyro.conf.arg(metavar='FILE')]

    def run(self) -> int:
        """Write the table and return the exit code."""
        context = _context_of(
            study=self.study, machine=self.machine, grid=self.grid, only=self.only, read_only=True
        )
        _warn_no_runs(context=context)
        return results(context=context, out=self.out)


STUDY_COMMANDS = {'submit': Submit, 'status': Status, 'resubmit': Resubmit, 'results': Results}

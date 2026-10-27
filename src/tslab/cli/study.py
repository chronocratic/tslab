"""The ``tslab study`` subcommands: thin tyro dataclasses over ``tslab.study.commands``.

The argument dataclasses here hold only the flags of one command; the
behavior lives in ``tslab.study.commands``, which the ``run`` methods call.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import tyro

from tslab.cli.usage import UsageError
from tslab.study.commands import parse_states, resubmit, results, status, submit
from tslab.study.context import Context, load_context, Selection
from tslab.study.machine import resolve_machine
from tslab.study.state import StudyState

__all__ = ['STUDY_COMMANDS', 'Resubmit', 'Results', 'Status', 'Submit']


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
    except (ValueError, TypeError, FileNotFoundError, ModuleNotFoundError) as error:
        raise UsageError(str(error)) from error


def _states_of(*, values: tuple[str, ...]) -> tuple[StudyState, ...]:
    try:
        return parse_states(values=values)
    except ValueError as error:
        raise UsageError(str(error)) from error


@dataclass(frozen=True, kw_only=True)
class Submit:
    """Submit the runs of a study to a machine, or dry-run the plan.

    For example:

        tslab study submit my_paper.studies:example --machine local --dry-run
        tslab study submit my_paper.studies:example --machine local -- --trainer.max-epochs 5

    Attributes:
        study: The study reference, module:attribute.
        machine: The machine name (machines/<name>.toml above the working directory, or
            registered with tslab machine add) or a profile path.
        grid: Keep only this grid; repeatable.
        only: Keep only the runs that match axis=value; repeatable,
            comma separated for several values of one axis.
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
        return submit(
            context=_context_of(
                study=self.study, machine=self.machine, grid=self.grid, only=self.only
            ),
            include_done=self.include_done,
            dry_run=self.dry_run,
            extra_args=self.extra_args,
        )


@dataclass(frozen=True, kw_only=True)
class Status:
    """The state table of a study, and one line per run not done.

    For example:

        tslab study status my_paper.studies:example --machine local

    Attributes:
        study: The study reference, module:attribute.
        machine: The machine name (machines/<name>.toml above the working directory, or
            registered with tslab machine add) or a profile path.
        grid: Keep only this grid; repeatable.
        only: Keep only the runs that match axis=value; repeatable,
            comma separated for several values of one axis.
        states: Show only the runs in these states; repeatable, comma separated.
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

    def run(self) -> int:
        """Print the table and return the exit code."""
        states = _states_of(values=self.states)
        return status(
            context=_context_of(
                study=self.study,
                machine=self.machine,
                grid=self.grid,
                only=self.only,
                read_only=True,
            ),
            states=states,
        )


@dataclass(frozen=True, kw_only=True)
class Resubmit:
    """Submit the interrupted, missing or failed runs of a study again.

    For example:

        tslab study resubmit my_paper.studies:example --machine local
        tslab study resubmit my_paper.studies:example --machine local --dry-run

    Attributes:
        study: The study reference, module:attribute.
        machine: The machine name (machines/<name>.toml above the working directory, or
            registered with tslab machine add) or a profile path.
        grid: Keep only this grid; repeatable.
        only: Keep only the runs that match axis=value; repeatable,
            comma separated for several values of one axis.
        states: The states to resubmit; repeatable, comma separated.
            Interrupted and missing when omitted.
        cancel_live: Cancel the queued and running runs first and
            resubmit them a little later.
        dry_run: Write the job files and print the plan, submit nothing.
        auto: Follow-up mode, used by TSLab itself.
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
        return resubmit(
            context=_context_of(
                study=self.study, machine=self.machine, grid=self.grid, only=self.only
            ),
            states=states,
            cancel_live=self.cancel_live,
            dry_run=self.dry_run,
            extra_args=self.extra_args or None,
            auto=self.auto,
        )


@dataclass(frozen=True, kw_only=True)
class Results:
    """The metrics of every run of a study, as one CSV table.

    For example:

        tslab study results my_paper.studies:example --machine local --out results.csv

    Attributes:
        study: The study reference, module:attribute.
        machine: The machine name (machines/<name>.toml above the working directory, or
            registered with tslab machine add) or a profile path.
        grid: Keep only this grid; repeatable.
        only: Keep only the runs that match axis=value; repeatable,
            comma separated for several values of one axis.
        out: The CSV file to write; required.
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
        return results(
            context=_context_of(
                study=self.study,
                machine=self.machine,
                grid=self.grid,
                only=self.only,
                read_only=True,
            ),
            out=self.out,
        )


STUDY_COMMANDS = {'submit': Submit, 'status': Status, 'resubmit': Resubmit, 'results': Results}

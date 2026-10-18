"""The ``tslab study`` subcommands: thin tyro dataclasses over ``tslab.study.commands``.

The argument dataclasses here hold only the flags of one command; the
behavior lives in ``tslab.study.commands``, which the ``run`` methods call.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Annotated

import tyro

from tslab.study import load_context, parse_states, Selection, status, submit

__all__ = ['STUDY_COMMANDS', 'Status', 'Submit']

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class Submit:
    """Submit the runs of a study to a machine, or dry-run the plan.

    Attributes:
        study: The study reference, ``module:attribute``.
        machine: The machine profile to submit to.
        grid: Keep only this grid; repeatable.
        only: Keep only the runs that match ``axis=value``; repeatable,
            comma separated for several values of one axis.
        include_done: Submit the done runs as well.
        dry_run: Write the job files and print the plan, submit nothing.
        extra_args: The arguments after ``--``, appended to every run.
    """

    study: Annotated[str, tyro.conf.Positional]
    machine: Path
    grid: Annotated[tuple[str, ...], tyro.conf.UseAppendAction] = ()
    only: Annotated[tuple[str, ...], tyro.conf.UseAppendAction] = ()
    include_done: bool = False
    dry_run: bool = False
    extra_args: tyro.conf.Suppress[tuple[str, ...]] = ()

    def run(self) -> int:
        """Submit and return the exit code."""
        return submit(
            context=load_context(
                reference=self.study,
                machine_path=self.machine,
                selection=Selection(grids=self.grid, only=self.only),
                project_root=Path.cwd(),
            ),
            include_done=self.include_done,
            dry_run=self.dry_run,
            extra_args=self.extra_args,
        )


@dataclass(frozen=True, kw_only=True)
class Status:
    """The state table of a study, and one line per run not done.

    Attributes:
        study: The study reference, ``module:attribute``.
        machine: The machine profile the runs run on.
        grid: Keep only this grid; repeatable.
        only: Keep only the runs that match ``axis=value``; repeatable,
            comma separated for several values of one axis.
        state: Show only the runs in this state; repeatable, comma separated
            for several states.
    """

    study: Annotated[str, tyro.conf.Positional]
    machine: Path
    grid: Annotated[tuple[str, ...], tyro.conf.UseAppendAction] = ()
    only: Annotated[tuple[str, ...], tyro.conf.UseAppendAction] = ()
    state: Annotated[tuple[str, ...], tyro.conf.UseAppendAction] = ()

    def run(self) -> int:
        """Print the table and return the exit code."""
        try:
            states = parse_states(values=self.state)
        except ValueError as error:
            failure = error  # logged below, without a traceback: it is a usage error
        else:
            return status(
                context=load_context(
                    reference=self.study,
                    machine_path=self.machine,
                    selection=Selection(grids=self.grid, only=self.only),
                    project_root=Path.cwd(),
                ),
                states=states,
            )
        _LOGGER.error('%s', failure)
        return 2


STUDY_COMMANDS = {'submit': Submit, 'status': Status}

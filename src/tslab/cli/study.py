"""The ``tslab study`` subcommands: thin tyro dataclasses over ``tslab.study.commands``.

The argument dataclasses here hold only the flags of one command; the
behavior lives in ``tslab.study.commands``, which the ``run`` methods call.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import tyro

from tslab.study import load_context, Selection, submit

__all__ = ['STUDY_COMMANDS', 'Submit']


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


STUDY_COMMANDS: dict[str, type[Submit]] = {'submit': Submit}

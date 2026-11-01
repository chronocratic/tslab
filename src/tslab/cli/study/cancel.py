"""The ``tslab study cancel`` command."""

from dataclasses import dataclass
import logging
from typing import Annotated

import tyro

from tslab.cli.common import print_json
from tslab.cli.study import _context_of, _selection_args
from tslab.study.commands import cancel
from tslab.study.documents import cancel_document

__all__ = ['Cancel']

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class Cancel:
    """Cancel queued and running SLURM jobs, or stop the live local launch.

    For example:
        tslab study cancel my_paper.studies:example --machine cluster

    Attributes:
        study: The study reference, module:attribute.
        machine: A machine name or profile path.
        grid: Keep only this grid.
        only: Keep only runs that match axis=value.
        dry_run: Show job ids without cancelling them.
        signal: Send USR2 to the running jobs instead of cancelling them, to test
            that runs save and resubmit; the jobs then end on their own.
        json: Print one study-cancel JSON document.
    """

    study: Annotated[str, tyro.conf.arg(metavar='STUDY'), tyro.conf.Positional]
    machine: Annotated[str, tyro.conf.arg(metavar='MACHINE')]
    grid: Annotated[tuple[str, ...], tyro.conf.arg(metavar='GRID'), tyro.conf.UseAppendAction] = ()
    only: Annotated[
        tuple[str, ...], tyro.conf.arg(metavar='AXIS=VALUES'), tyro.conf.UseAppendAction
    ] = ()
    dry_run: bool = False
    signal: bool = False
    json: bool = False

    def run(self) -> int:
        """Cancel selected jobs and return the process exit code."""
        context = _context_of(
            study=self.study, machine=self.machine, grid=self.grid, only=self.only
        )
        signal = 'USR2' if self.signal else None
        code, cancelled = cancel(context=context, dry_run=self.dry_run, signal=signal)
        if code == 0 and cancelled and not self.dry_run:
            selection = _selection_args(
                study=self.study, machine=self.machine, grid=self.grid, only=self.only
            )
            _LOGGER.info('follow progress: tslab study status %s', selection)
        if self.json:
            print_json(
                document=cancel_document(
                    cancelled=cancelled, dry_run=self.dry_run, exit_code=code, signal=signal
                )
            )
        return code

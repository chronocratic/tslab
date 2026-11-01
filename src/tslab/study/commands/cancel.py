"""The cancel command: cancel or signal the live jobs of selected runs."""

import logging

from tslab.study.context import Context
from tslab.study.launchers import backend_for, LaunchError

__all__ = ['cancel']

_LOGGER = logging.getLogger(__name__)


def cancel(*, context: Context, dry_run: bool, signal: str | None = None) -> tuple[int, list[str]]:
    """Cancel or signal the queued and running SLURM jobs of selected runs.

    On a local machine it stops the live local launch with a SIGTERM; the
    selection must cover every live task, because the launch is one unit.

    Args:
        context: The study context containing selected run reports.
        dry_run: Log the job ids without cancelling or signalling them.
        signal: Send this signal to the running jobs instead of cancelling
            them, so the runs save and end on their own; queued runs are
            skipped because a pending job has no step to signal.

    Returns:
        The process exit code and the job ids cancelled, signalled, or
        selected for either.
    """
    try:
        return backend_for(machine=context.machine).cancel(
            context=context, dry_run=dry_run, signal=signal
        )
    except LaunchError as error:
        _LOGGER.error('%s', error)  # noqa: TRY400 # why: cancellation failures need no traceback
        return 1, []

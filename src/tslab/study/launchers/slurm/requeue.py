"""Requeue of a chained SLURM task in place."""

import logging
import os

from tslab.jobs.slurm import scontrol_requeue, SlurmError

__all__ = ['requeue_task', 'restarted_attempt']

_LOGGER = logging.getLogger(__name__)


def restarted_attempt(*, attempt: int) -> int:
    """The attempt number of a requeued task: the recorded one plus the restarts."""
    return attempt + int(os.environ.get('SLURM_RESTART_COUNT', '0'))


def requeue_task(*, task_id: str) -> bool:
    """Requeue one array task; log and return ``False`` when the scheduler refuses.

    Args:
        task_id: ``<array job id>_<task id>``.

    Returns:
        Whether the requeue call succeeded.
    """
    try:
        scontrol_requeue(task_id=task_id)
    except SlurmError as error:
        _LOGGER.error('%s', error)  # noqa: TRY400 # why: a failed requeue needs no traceback
        return False
    return True

"""Requeue of a chained SLURM task in place."""

import logging

from tslab.jobs.slurm import scontrol_requeue, SlurmError

__all__ = ['at_requeue_limit', 'requeue_task', 'restarted_attempt']

_LOGGER = logging.getLogger(__name__)


def restarted_attempt(*, attempt: int, restart_count: int) -> int:
    """The attempt number of a requeued task: its record plus scheduler restarts.

    Args:
        attempt: The attempt number in the latest task record.
        restart_count: The ``SLURM_RESTART_COUNT`` value read by the caller.

    Returns:
        The attempt number including unrecorded scheduler restarts.
    """
    return attempt + restart_count


def at_requeue_limit(*, restart_count: int, max_requeues: int | None) -> bool:
    """Whether one more requeue would reach ``MaxBatchRequeue``.

    Args:
        restart_count: The restart count read by the caller.
        max_requeues: The parsed scheduler limit, or ``None`` when unavailable.

    Returns:
        Whether the scheduler will hold the job after another requeue.
    """
    if max_requeues is None:
        return False
    return restart_count + 1 >= max_requeues


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

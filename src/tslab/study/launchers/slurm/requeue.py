"""Requeue of a chained SLURM task in place."""

import logging
import os

from tslab.jobs.slurm import query_cluster_config, scontrol_requeue, SlurmError

__all__ = ['at_requeue_limit', 'requeue_task', 'restarted_attempt']

_LOGGER = logging.getLogger(__name__)


def restarted_attempt(*, attempt: int) -> int:
    """The attempt number of a requeued task: the recorded one plus the restarts."""
    return attempt + int(os.environ.get('SLURM_RESTART_COUNT', '0'))


def at_requeue_limit() -> bool:
    """Whether one more requeue would reach ``MaxBatchRequeue``, where SLURM holds the job."""
    try:
        limit = int(query_cluster_config().get('MaxBatchRequeue', ''))
    except (SlurmError, ValueError):
        return False
    return restarted_attempt(attempt=0) + 1 >= limit


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

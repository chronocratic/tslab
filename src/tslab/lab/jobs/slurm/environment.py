"""The environment of the SLURM job this process runs in."""

from collections.abc import Mapping

__all__ = ['RESTART_COUNT_VARIABLE', 'task_count']

RESTART_COUNT_VARIABLE = 'SLURM_RESTART_COUNT'


def task_count(*, environment: Mapping[str, str]) -> int:
    """Return the number of tasks in the SLURM job, or one outside SLURM.

    Args:
        environment: Environment variables, usually ``os.environ``.

    Returns:
        ``SLURM_NTASKS`` as a number, or ``1`` when it is not set.
    """
    return int(environment.get('SLURM_NTASKS', '1'))

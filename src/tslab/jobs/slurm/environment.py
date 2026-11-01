"""The environment of the SLURM job this process runs in."""

from collections.abc import Mapping

__all__ = ['task_count']


def task_count(*, environment: Mapping[str, str]) -> int:
    """Return the number of tasks in the SLURM job, or one outside SLURM.

    Args:
        environment: Environment variables, usually ``os.environ``.
    """
    return int(environment.get('SLURM_NTASKS', '1'))

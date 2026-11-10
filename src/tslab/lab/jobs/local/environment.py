"""The environment a local task runs in."""

from collections.abc import Mapping

__all__ = ['without_scheduler_variables']


def without_scheduler_variables(*, environ: Mapping[str, str]) -> dict[str, str]:
    """A copy of the environment without the ``SLURM_*`` variables.

    A local launch started inside a SLURM allocation must not hand the allocation
    (task count, rank, port) to the tasks it starts itself.

    Args:
        environ: The environment to filter, usually ``os.environ``.

    Returns:
        The variables whose name does not start with ``SLURM_``.
    """
    return {name: value for name, value in environ.items() if not name.startswith('SLURM_')}

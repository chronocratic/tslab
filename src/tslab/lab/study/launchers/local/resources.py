"""CPU availability for a local launch."""

from collections.abc import Callable

__all__ = ['usable_cpus']


def usable_cpus(*, affinity: Callable[[], int]) -> int:
    """The CPUs a process may run on, from an affinity query."""
    return max(affinity(), 1)

"""The CPUs this process may run on."""

import os
import sys

from tslab.study.launchers.local.resources import usable_cpus

__all__ = ['available_cpus']


def available_cpus() -> int:
    """The CPUs this process may run on; a local launch runs its tasks here."""
    # why: the affinity mask honours taskset and container limits; macOS has no such call
    if sys.platform == 'linux':
        return usable_cpus(affinity=lambda: len(os.sched_getaffinity(0)))
    return usable_cpus(affinity=lambda: os.cpu_count() or 1)

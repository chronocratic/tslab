"""The launch backends: one object per launcher, SLURM and local apart.

A study command asks the backend of the machine it runs on, picked once by
``backend_for`` from the profile's launcher; it never names a launcher. The
backend code lives in the ``slurm`` and ``local`` subpackages, which never
import each other; what they share lives in ``common``.
"""

from collections.abc import Mapping
import os
import sys

from tslab.enums import LauncherKind
from tslab.study.launchers.common.protocols import Backend
from tslab.study.launchers.local import LocalBackend
from tslab.study.launchers.local.records import LocalJobRecord
from tslab.study.launchers.local.resources import usable_cpus
from tslab.study.launchers.slurm import SlurmBackend
from tslab.study.launchers.slurm.records import SlurmJobRecord
from tslab.study.machine import MachineProfile
from tslab.study.records import JobRecord

__all__ = ['BACKENDS', 'JOB_RECORDS', 'available_cpus', 'backend_for']

#: One backend per launcher; the only place beside the machine tables that
#: names both.
BACKENDS: Mapping[LauncherKind, Backend] = {
    LauncherKind.SLURM: SlurmBackend(),
    LauncherKind.LOCAL: LocalBackend(),
}

#: The job record class of each launcher; ``JobRecord.from_json`` reads it.
JOB_RECORDS: Mapping[LauncherKind, type[JobRecord]] = {
    LauncherKind.SLURM: SlurmJobRecord,
    LauncherKind.LOCAL: LocalJobRecord,
}


def backend_for(*, machine: MachineProfile) -> Backend:
    """The backend that serves the machine's launcher.

    Args:
        machine: The machine profile to read the launcher from.

    Returns:
        The backend object of the machine's launcher.
    """
    backend = BACKENDS.get(machine.launcher)
    if backend is None:
        message = f'no backend serves launcher {machine.launcher!r}; check the machine profile'
        raise ValueError(message)
    return backend


def available_cpus() -> int:
    """The CPUs this process may run on; a local launch runs its tasks here."""
    # why: the affinity mask honours taskset and container limits; macOS has no such call
    if sys.platform == 'linux':
        return usable_cpus(affinity=lambda: len(os.sched_getaffinity(0)))
    return usable_cpus(affinity=lambda: os.cpu_count() or 1)

"""The launcher tables: the backend and the job record class of each launcher.

The only module of the launchers that names both backends.
"""

from collections.abc import Mapping

from tslab.enums import LauncherKind
from tslab.study.launchers.common.protocols import Backend
from tslab.study.launchers.local import LocalBackend
from tslab.study.launchers.local.records import LocalJobRecord
from tslab.study.launchers.slurm import SlurmBackend
from tslab.study.launchers.slurm.records import SlurmJobRecord
from tslab.study.machine import MachineProfile
from tslab.study.records import JobRecord

__all__ = ['BACKENDS', 'JOB_RECORDS', 'backend_for']

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

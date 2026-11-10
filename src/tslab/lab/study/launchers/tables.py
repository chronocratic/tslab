"""The launcher tables: the backend and the job record class of each launcher.

The only module of the launchers that names both backends.
"""

from collections.abc import Mapping

from tslab.lab.enums import LauncherKind
from tslab.lab.study.launchers.common.protocols import Backend
from tslab.lab.study.launchers.local import LocalBackend
from tslab.lab.study.launchers.local.records import LocalJobRecord
from tslab.lab.study.launchers.slurm import SlurmBackend
from tslab.lab.study.launchers.slurm.records import SlurmJobRecord
from tslab.lab.study.machine import MachineProfile
from tslab.lab.study.records import JobRecord

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

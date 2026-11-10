"""Machine profiles: one TOML file per machine, read strictly.

A profile names the launcher, the output root, settings and environment for
every run on that machine, the SLURM resources and overrides matched by axis
values. Unknown keys raise: a typo in a profile must fail at submit, not
silently change a cluster job.
"""

from tslab.lab.study.machine.load import load_machine
from tslab.lab.study.machine.local import LocalSettings, timeout_seconds
from tslab.lab.study.machine.profile import FollowupSettings, MachineOverride, MachineProfile
from tslab.lab.study.machine.register import (
    MachineLocation,
    read_register,
    register_file,
    resolve_machine,
    validate_machine_name,
    write_register,
)
from tslab.lab.study.machine.slurm import SlurmSettings
from tslab.lab.study.machine.tables import flatten_settings, MachineProfileError

__all__ = [
    'FollowupSettings',
    'LocalSettings',
    'MachineLocation',
    'MachineOverride',
    'MachineProfile',
    'MachineProfileError',
    'SlurmSettings',
    'flatten_settings',
    'load_machine',
    'read_register',
    'register_file',
    'resolve_machine',
    'timeout_seconds',
    'validate_machine_name',
    'write_register',
]

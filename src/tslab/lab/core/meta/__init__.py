"""Metadata written next to runs and instances: the ``meta.json`` files."""

from tslab.lab.core.meta.instance import (
    can_resume,
    describe_external_inputs,
    describe_inputs,
    identity_from_meta,
    instance_meta_document,
    read_instance_meta,
    stale_reason,
)
from tslab.lab.core.meta.run import (
    CODE_SNAPSHOT_VARIABLE,
    CODE_SNAPSHOT_VARIABLES,
    collect_code_snapshot,
    collect_git_state,
    collect_hardware,
    collect_versions,
    collectors_generation,
    hostname,
    package_version,
    register_hardware,
    register_versions,
)

__all__ = [
    'CODE_SNAPSHOT_VARIABLE',
    'CODE_SNAPSHOT_VARIABLES',
    'can_resume',
    'collect_code_snapshot',
    'collect_git_state',
    'collect_hardware',
    'collect_versions',
    'collectors_generation',
    'describe_external_inputs',
    'describe_inputs',
    'hostname',
    'identity_from_meta',
    'instance_meta_document',
    'package_version',
    'read_instance_meta',
    'register_hardware',
    'register_versions',
    'stale_reason',
]

"""Local helpers for job processes."""

from tslab.lab.jobs.local.background import start_detached, start_keep_awake, stop_keep_awake
from tslab.lab.jobs.local.environment import without_scheduler_variables
from tslab.lab.jobs.local.gpus import (
    busy_gpus,
    gpu_lock_folder,
    GpuDiscovery,
    GpuLease,
    MPS_GPU,
    select_gpus,
    try_lease_gpu,
    visible_gpus,
)
from tslab.lab.jobs.local.output import tail_lines, TaskInfo, TaskOutput, TaskOutputState
from tslab.lab.jobs.local.pool import (
    OnEnd,
    OnEvent,
    OnLine,
    OnSpawn,
    OnStart,
    OnWait,
    PoolCommand,
    run_commands,
)
from tslab.lab.jobs.local.process import (
    boot_id,
    process_alive,
    ProcessResult,
    run_process,
    stop_on_signals,
    StopRequest,
)
from tslab.lab.jobs.local.shell import export_lines
from tslab.lab.jobs.local.watchdog import Watchdog

__all__ = [
    'MPS_GPU',
    'GpuDiscovery',
    'GpuLease',
    'OnEnd',
    'OnEvent',
    'OnLine',
    'OnSpawn',
    'OnStart',
    'OnWait',
    'PoolCommand',
    'ProcessResult',
    'StopRequest',
    'TaskInfo',
    'TaskOutput',
    'TaskOutputState',
    'Watchdog',
    'boot_id',
    'busy_gpus',
    'export_lines',
    'gpu_lock_folder',
    'process_alive',
    'run_commands',
    'run_process',
    'select_gpus',
    'start_detached',
    'start_keep_awake',
    'stop_keep_awake',
    'stop_on_signals',
    'tail_lines',
    'try_lease_gpu',
    'visible_gpus',
    'without_scheduler_variables',
]

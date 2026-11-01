"""Local helpers for job processes."""

from tslab.jobs.local.background import start_detached, start_keep_awake, stop_keep_awake
from tslab.jobs.local.gpus import (
    busy_gpus,
    gpu_lock_folder,
    GpuLease,
    select_gpus,
    try_lease_gpu,
    visible_gpus,
)
from tslab.jobs.local.output import tail_lines, TaskInfo, TaskOutput
from tslab.jobs.local.pool import OnEnd, OnEvent, OnLine, OnStart, OnWait, PoolCommand, run_commands
from tslab.jobs.local.process import (
    process_alive,
    ProcessResult,
    run_process,
    stop_on_signals,
    StopRequest,
)

__all__ = [
    'GpuLease',
    'OnEnd',
    'OnEvent',
    'OnLine',
    'OnStart',
    'OnWait',
    'PoolCommand',
    'ProcessResult',
    'StopRequest',
    'TaskInfo',
    'TaskOutput',
    'busy_gpus',
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
]

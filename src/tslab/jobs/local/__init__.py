"""Local helpers for job processes."""

from tslab.jobs.local.background import start_detached, start_keep_awake, stop_keep_awake
from tslab.jobs.local.process import ProcessResult, run_process, stop_on_signals, StopRequest

__all__ = [
    'ProcessResult',
    'StopRequest',
    'run_process',
    'start_detached',
    'start_keep_awake',
    'stop_keep_awake',
    'stop_on_signals',
]

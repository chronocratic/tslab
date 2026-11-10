"""Cancelling a live local launch."""

import logging
import os
import signal as signal_module

from tslab.lab.core.meta import hostname
from tslab.lab.enums import LocalTaskState
from tslab.lab.jobs.local import boot_id
from tslab.lab.study.context import Context
from tslab.lab.study.launchers.local.pid_file import launch_is_live, LocalLaunch, read_launch
from tslab.lab.study.paths import study_folder

__all__ = ['cancel_local']

_LOGGER = logging.getLogger(__name__)


def cancel_local(*, context: Context, dry_run: bool, signal: str | None) -> tuple[int, list[str]]:
    """Stop the local launch, live or dead, when the selection covers every live task."""
    if signal is not None:
        _LOGGER.error('cancel --signal is SLURM only')
        return 2, []
    launch = read_launch(study_folder=study_folder(machine=context.machine, study=context.study))
    if launch is None:
        _LOGGER.info('nothing to cancel')
        return 0, []
    selected = {report.job_id for report in context.reports}
    # why: the tasks of a group that has not started have no job record, so no selection names them
    recorded = {f'{job.job_id}_{task.index}' for job in context.jobs for task in job.tasks}
    missed = [
        key
        for key, state in launch.tasks
        if state != LocalTaskState.ENDED and key in recorded and key not in selected
    ]
    if missed:
        _LOGGER.error(
            'a local launch is cancelled as one unit; the selection misses its tasks: %s',
            ', '.join(missed),
        )
        return 2, []
    if not launch_is_live(launch=launch, host=hostname()):
        return _cancel_dead(launch=launch, dry_run=dry_run)
    if dry_run:
        _LOGGER.info('would stop local launch %s', launch.pid)
        return 0, [str(launch.pid)]
    os.kill(launch.pid, signal_module.SIGTERM)
    _LOGGER.info('sent SIGTERM to local launch %s', launch.pid)
    return 0, [str(launch.pid)]


def _cancel_dead(*, launch: LocalLaunch, dry_run: bool) -> tuple[int, list[str]]:
    """Stop the runs of a launcher that is gone, when the file is from this boot.

    A group that exits between the existence check and the signal is ignored,
    and the loop signals the rest; the returned pids are the groups the loop
    tried to signal.
    """
    if launch.hostname != hostname():
        _LOGGER.info('nothing to cancel')
        return 0, []
    current = boot_id()
    if launch.boot is not None and current is not None and launch.boot != current:
        _LOGGER.info('nothing to cancel')
        return 0, []
    states = dict(launch.tasks)
    pids = [
        pid
        for task_id, pid in launch.run_pids
        if states.get(task_id) != LocalTaskState.ENDED and _group_exists(pid=pid)
    ]
    if not pids:
        _LOGGER.info('nothing to cancel')
        return 0, []
    pids_text = ', '.join(str(pid) for pid in pids)
    if dry_run:
        _LOGGER.info('the launcher of pid %s is gone; its runs: %s', launch.pid, pids_text)
        return 0, [str(pid) for pid in pids]
    for pid in pids:
        try:
            os.killpg(pid, signal_module.SIGTERM)
        except (ProcessLookupError, PermissionError):
            # why: the group may have exited between the existence check and the signal
            _LOGGER.debug('the group of pid %s is already gone; skipping it', pid)
    _LOGGER.info(
        'the launcher of pid %s is gone; sent SIGTERM to its runs: %s', launch.pid, pids_text
    )
    return 0, [str(pid) for pid in pids]


def _group_exists(*, pid: int) -> bool:
    """Whether the process group of ``pid`` still exists and is ours."""
    try:
        os.killpg(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True

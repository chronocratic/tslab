"""Cancelling a live local launch."""

import logging
import os
import signal as signal_module

from tslab.core.meta import hostname
from tslab.study.context import Context
from tslab.study.launchers.local.pid_file import launch_is_live, read_launch
from tslab.study.paths import study_folder

__all__ = ['cancel_local']

_LOGGER = logging.getLogger(__name__)


def cancel_local(*, context: Context, dry_run: bool, signal: str | None) -> tuple[int, list[str]]:
    """Stop the live local launch when selection covers every live task."""
    if signal is not None:
        _LOGGER.error('cancel --signal is SLURM only')
        return 2, []
    launch = read_launch(study_folder=study_folder(machine=context.machine, study=context.study))
    if launch is None or not launch_is_live(launch=launch, host=hostname()):
        _LOGGER.info('nothing to cancel')
        return 0, []
    selected = {report.job_id for report in context.reports}
    missed = [key for key, state in launch.tasks if state != 'ended' and key not in selected]
    if missed:
        _LOGGER.error(
            'a local launch is cancelled as one unit; the selection misses its tasks: %s',
            ', '.join(missed),
        )
        return 2, []
    if dry_run:
        _LOGGER.info('would stop local launch %s', launch.pid)
        return 0, [str(launch.pid)]
    os.kill(launch.pid, signal_module.SIGTERM)
    _LOGGER.info('sent SIGTERM to local launch %s', launch.pid)
    return 0, [str(launch.pid)]

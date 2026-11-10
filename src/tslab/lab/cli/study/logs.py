"""The ``tslab study logs`` command."""

from collections import deque
from dataclasses import dataclass
import logging
from pathlib import Path
import sys
import time
from typing import Annotated, cast

import tyro

from tslab.cli.common import resolve_request
from tslab.cli.study.shared import _context_of
from tslab.core.meta import hostname
from tslab.core.paths import LOGS_DIRECTORY
from tslab.study.context import Context
from tslab.study.launchers.local.pid_file import LAUNCH_LOGS
from tslab.study.paths import study_folder

__all__ = ['Logs']

_LOGGER = logging.getLogger(__name__)
_POLL_SECONDS = 0.5
# why: the tests replace it to stop the follow loop
_sleep = time.sleep


@dataclass(frozen=True, kw_only=True)
class Logs:
    """Print the run log of one run, or follow it as it grows.

    For example:
        tslab study logs example --machine local --only seed=1 -f

    Attributes:
        study: The short study name or full module:attribute reference. A short name expands
            against the studies package of tslab.toml.
        machine: A machine name or profile path. Default: from
            [machine_by_host] in tslab.toml.
        grid: Keep only this grid.
        only: Keep only runs that match axis=value; the selection must match one run.
        follow: Keep printing new lines until Ctrl-C.
        lines: Start from the last this many lines.
        job: Show the scheduler output of the job instead of the run log.
        launch: Show the log of the latest local launch started with --detach instead.
    """

    study: Annotated[str, tyro.conf.arg(metavar='STUDY'), tyro.conf.Positional]
    machine: Annotated[str | None, tyro.conf.arg(metavar='MACHINE', help_behavior_hint='')] = None
    grid: Annotated[tuple[str, ...], tyro.conf.arg(metavar='GRID'), tyro.conf.UseAppendAction] = ()
    only: Annotated[
        tuple[str, ...], tyro.conf.arg(metavar='AXIS=VALUES'), tyro.conf.UseAppendAction
    ] = ()
    follow: Annotated[bool, tyro.conf.arg(aliases=('-f',))] = False
    lines: Annotated[int, tyro.conf.arg(aliases=('-n',), metavar='N')] = 50
    job: bool = False
    launch: bool = False

    def run(self) -> int:
        """Print the log and return the process exit code."""
        request = resolve_request(
            study=self.study, machine=self.machine, start=Path.cwd(), host=hostname()
        )
        context = _context_of(
            study=cast('str', request.study),
            machine=cast('str', request.machine),
            grid=self.grid,
            only=self.only,
            read_only=True,
        )
        path = self._launch_log(context=context) if self.launch else self._run_log(context=context)
        if path is None:
            return 2
        try:
            return _print_log(path=path, lines=self.lines, follow=self.follow)
        except KeyboardInterrupt:
            return 0

    def _launch_log(self, *, context: Context) -> Path | None:
        folder = study_folder(machine=context.machine, study=context.study) / LAUNCH_LOGS
        paths = list(folder.glob('*.log'))
        if paths:
            return max(paths, key=lambda path: path.stat().st_mtime)
        _LOGGER.error('no launch log in %s; only tslab study submit --detach writes one', folder)
        return None

    def _run_log(self, *, context: Context) -> Path | None:
        reports = context.reports
        if len(reports) != 1:
            paths = ''.join(f'\n  {report.run.run_path}' for report in reports)
            _LOGGER.error(
                'the selection matches %d runs; narrow it with --only:%s', len(reports), paths
            )
            return None
        report = reports[0]
        if not self.job:
            return context.study_directory / report.run.run_path / LOGS_DIRECTORY / 'run.log'
        if report.task is None:
            _LOGGER.error('%s was never submitted, so it has no job output', report.run.run_path)
            return None
        return context.log_file(run=report.run, index=report.task.index)


def _write(data: bytes) -> None:
    sys.stdout.write(data.decode('utf-8', errors='replace'))
    sys.stdout.flush()


def _read_from(*, path: Path, position: int) -> bytes:
    """The bytes of the file after ``position``; empty when it is missing."""
    try:
        with path.open('rb') as file:
            file.seek(position)
            return file.read()
    except FileNotFoundError:
        return b''


def _print_log(*, path: Path, lines: int, follow: bool) -> int:
    """Print the last lines of the file, then, with ``follow``, new data until Ctrl-C."""
    while not path.exists() and follow:
        _sleep(_POLL_SECONDS)
    if not path.exists():
        _LOGGER.info('no log yet at %s', path)
        return 0
    with path.open('rb') as file:
        _write(b''.join(deque(file, maxlen=lines)))
        position = file.tell()
    while follow:
        _sleep(_POLL_SECONDS)
        if path.exists() and path.stat().st_size < position:
            # why: a smaller file means the run started over (a resubmit); read it from the top
            position = 0
        data = _read_from(path=path, position=position)
        _write(data)
        position += len(data)
    return 0

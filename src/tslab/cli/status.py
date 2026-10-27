"""``tslab status``: what the instances of a run, or of every run below a folder, are doing.

Given a run folder (one holding ``status.json``), it prints one line per stage
instance and the counts per stage. Given any other folder, it finds every run
below it and prints one line per run, so the failed runs of a whole study show
on one screen.

A run whose ``status.json`` still says ``running`` but whose lock has no recent
heartbeat was killed (for example at a SLURM time limit); it is shown as
``interrupted``. Running the same script again resumes it.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Annotated

import tyro

from tslab.core.paths import STATUS_FILE
from tslab.core.run import is_run_active
from tslab.core.status import (
    InstanceRecord,
    InstanceState,
    read_status,
    RunState,
    RunStatusSnapshot,
    StatusFileError,
)
from tslab.core.text import format_table

__all__ = ['Status']

_LOGGER = logging.getLogger(__name__)
_INTERRUPTED = 'interrupted'
_FINGERPRINT_PREFIX = 8
_SECONDS_PER_MINUTE = 60
_SECONDS_PER_HOUR = 3600
_COUNTED_STATES = (
    InstanceState.DONE,
    InstanceState.FAILED,
    InstanceState.SKIPPED,
    InstanceState.RUNNING,
)


@dataclass(frozen=True, kw_only=True)
class Status:
    """Show the instances of a run, or one line per run below a folder.

    For example:

        tslab status runs/
        tslab status runs/ --stage scale

    Attributes:
        path: A run folder, or any folder above runs (for example runs/).
        stage: Count only the instances of this stage.
    """

    path: Annotated[Path, tyro.conf.arg(metavar='FOLDER'), tyro.conf.Positional]
    stage: str | None = None

    def run(self) -> int:
        """Print the status.

        Returns:
            ``0`` when something was shown, ``1`` when no run was found or the
            status file of the given run is unreadable, ``2`` when ``path`` is
            not a folder.
        """
        if not self.path.is_dir():
            _LOGGER.error('%s is not a folder', self.path.as_posix())
            return 2
        try:
            snapshot = read_status(path=self.path / STATUS_FILE)
        except StatusFileError as error:
            _LOGGER.error('%s', error)  # noqa: TRY400 # why: a traceback adds nothing
            return 1
        if snapshot is not None:
            _print_run(directory=self.path, snapshot=snapshot, stage=self.stage)
            return 0
        runs = sorted(status_file.parent for status_file in self.path.rglob(STATUS_FILE))
        if not runs:
            _LOGGER.info('no runs below %s', self.path.as_posix())
            return 1
        _print_runs(root=self.path, runs=runs, stage=self.stage)
        return 0


def run_state_shown(*, directory: Path, snapshot: RunStatusSnapshot) -> str:
    """Return the run state to display: ``interrupted`` for a dead ``running`` run.

    Args:
        directory: The run folder.
        snapshot: Its ``status.json``.
    """
    if snapshot.run_state is RunState.RUNNING and not is_run_active(directory=directory):
        return _INTERRUPTED
    return snapshot.run_state.value


def _stage_of(*, instance_id: str) -> str:
    return instance_id.partition('[')[0]


def _selected(
    *, snapshot: RunStatusSnapshot, stage: str | None
) -> list[tuple[str, InstanceRecord]]:
    return [
        (instance_id, record)
        for instance_id, record in snapshot.instances.items()
        if stage is None or _stage_of(instance_id=instance_id) == stage
    ]


def _instance_state_shown(*, record: InstanceRecord, run_state: str) -> str:
    if record.state is InstanceState.RUNNING and run_state == _INTERRUPTED:
        return _INTERRUPTED
    return record.state.value


def _duration(*, seconds: float | None) -> str:
    if seconds is None:
        return ''
    whole = round(seconds)
    if whole < _SECONDS_PER_MINUTE:
        return f'{whole}s'
    if whole < _SECONDS_PER_HOUR:
        return f'{whole // _SECONDS_PER_MINUTE}m{whole % _SECONDS_PER_MINUTE:02d}s'
    minutes = (whole % _SECONDS_PER_HOUR) // _SECONDS_PER_MINUTE
    return f'{whole // _SECONDS_PER_HOUR}h{minutes:02d}m'


def _note(*, record: InstanceRecord) -> str:
    text = record.error or record.reason or ''
    return text.splitlines()[0] if text else ''


def _print_run(*, directory: Path, snapshot: RunStatusSnapshot, stage: str | None) -> None:
    run_state = run_state_shown(directory=directory, snapshot=snapshot)
    selected = _selected(snapshot=snapshot, stage=stage)
    rows = [
        (
            instance_id,
            _instance_state_shown(record=record, run_state=run_state),
            (record.fingerprint or '')[:_FINGERPRINT_PREFIX],
            _duration(seconds=record.seconds),
            _note(record=record),
        )
        for instance_id, record in selected
    ]
    _LOGGER.info('%s: %s', directory.as_posix(), run_state)
    for line in format_table(
        header=('instance', 'state', 'fingerprint', 'time', 'note'), rows=rows
    ):
        _LOGGER.info('%s', line)
    per_stage: dict[str, Counter[str]] = {}
    for instance_id, record in selected:
        counts = per_stage.setdefault(_stage_of(instance_id=instance_id), Counter())
        counts[_instance_state_shown(record=record, run_state=run_state)] += 1
    for stage_name, counts in per_stage.items():
        summary = ', '.join(f'{count} {state}' for state, count in sorted(counts.items()))
        _LOGGER.info('%s: %s', stage_name, summary)


def _print_runs(*, root: Path, runs: Sequence[Path], stage: str | None) -> None:
    rows: list[tuple[str, ...]] = []
    totals: Counter[str] = Counter()
    for directory in runs:
        try:
            snapshot = read_status(path=directory / STATUS_FILE)
        except StatusFileError as error:
            _LOGGER.warning('skipping an unreadable run: %s', error)
            continue
        if snapshot is None:  # removed while scanning
            continue
        run_state = run_state_shown(directory=directory, snapshot=snapshot)
        totals[run_state] += 1
        # Raw instance states: a dead run still shows its in-flight instances.
        counts = Counter(record.state for _, record in _selected(snapshot=snapshot, stage=stage))
        rows.append(
            (
                directory.relative_to(root).as_posix(),
                run_state,
                *(str(counts[state]) for state in _COUNTED_STATES),
            )
        )
    header = ('run', 'state', *(state.value for state in _COUNTED_STATES))
    for line in format_table(header=header, rows=rows):
        _LOGGER.info('%s', line)
    summary = ', '.join(f'{count} {state}' for state, count in sorted(totals.items()))
    _LOGGER.info('%d runs: %s', len(rows), summary)

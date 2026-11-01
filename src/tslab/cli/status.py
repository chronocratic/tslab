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
import shlex
from typing import Annotated

import tyro

from tslab.cli.common import print_json, UsageError, watch
from tslab.core.paths import STATUS_FILE
from tslab.core.status import (
    InstanceRecord,
    InstanceState,
    read_status,
    RunState,
    RunStatusSnapshot,
    shown_run_state,
    StatusFileError,
)
from tslab.core.text import log_table
from tslab.study.launchers import JobPhase
from tslab.study.reporting.outcome import states_exit_code
from tslab.study.reporting.state import read_folder, study_state, StudyState

__all__ = ['Status']

_LOGGER = logging.getLogger(__name__)
_INTERRUPTED = 'interrupted'
_FINGERPRINT_PREFIX = 8
_MAX_FAILED_INSTANCES = 10
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
        tslab status runs/ --watch 60

    Attributes:
        path: A run folder, or any folder above runs (for example runs/).
        stage: Count only the instances of this stage.
        watch: Refresh every this many seconds until Ctrl-C.
        json: Print one run-status JSON document.
        exit_code: Return the code for the run states.
    """

    path: Annotated[Path, tyro.conf.arg(metavar='FOLDER'), tyro.conf.Positional]
    stage: Annotated[str | None, tyro.conf.arg(metavar='STAGE', help_behavior_hint='')] = None
    watch: Annotated[float | None, tyro.conf.arg(metavar='SECONDS', help_behavior_hint='')] = None
    json: bool = False
    exit_code: bool = False

    def run(self) -> int:
        """Print the status.

        Returns:
            ``0`` when something was shown, ``1`` when no run was found or the
            status file of the given run is unreadable, ``2`` when ``path`` is
            not a folder.
        """
        if self.json and self.watch is not None:
            message = '--watch does not combine with --json'
            raise UsageError(message)
        if not self.path.is_dir():
            _LOGGER.error('%s is not a folder', self.path.as_posix())
            return 2
        return watch(seconds=self.watch, show=self._show)

    def _show(self) -> int:
        try:
            snapshot = read_status(path=self.path / STATUS_FILE)
        except StatusFileError as error:
            _LOGGER.error('%s', error)  # noqa: TRY400 # why: a traceback adds nothing
            return 1
        if snapshot is not None:
            _print_run(directory=self.path, snapshot=snapshot, stage=self.stage)
            if self.json or self.exit_code:
                entry, state = _run_status_entry(
                    directory=self.path, snapshot=snapshot, stage=self.stage
                )
                code = states_exit_code(states=[state], live=False)
                if self.json:
                    print_json(document=_status_document(entries=[entry], states=[state]))
                return code if self.exit_code else 0
            return 0
        runs = sorted(status_file.parent for status_file in self.path.rglob(STATUS_FILE))
        if not runs:
            _LOGGER.warning('no runs below %s', self.path.as_posix())
            _LOGGER.info('a run folder holds status.json; pass it, or a folder above runs')
            return 1
        _print_runs(root=self.path, runs=runs, stage=self.stage)
        if self.json or self.exit_code:
            entries, states = _run_status_entries(runs=runs, stage=self.stage)
            code = states_exit_code(states=states, live=False)
            if self.json:
                print_json(document=_status_document(entries=entries, states=states))
            return code if self.exit_code else 0
        return 0


def _run_status_entries(
    *, runs: Sequence[Path], stage: str | None
) -> tuple[list[dict[str, object]], list[StudyState]]:
    entries: list[dict[str, object]] = []
    states: list[StudyState] = []
    for directory in runs:
        try:
            snapshot = read_status(path=directory / STATUS_FILE)
        except StatusFileError:
            continue
        if snapshot is None:
            continue
        entry, state = _run_status_entry(directory=directory, snapshot=snapshot, stage=stage)
        entries.append(entry)
        states.append(state)
    return entries, states


def _run_status_entry(
    *, directory: Path, snapshot: RunStatusSnapshot, stage: str | None
) -> tuple[dict[str, object], StudyState]:
    folder = read_folder(run_directory=directory)
    state, _ = study_state(job_state=None, job_phase=JobPhase.UNKNOWN, folder=folder)
    run_state = shown_run_state(directory=directory, snapshot=snapshot)
    instances = [
        {
            'instance': instance_id,
            'stage': _stage_of(instance_id=instance_id),
            'state': _instance_state_shown(record=record, run_state=run_state),
            'fingerprint': record.fingerprint,
            'seconds': record.seconds,
            'error': record.error,
        }
        for instance_id, record in _selected(snapshot=snapshot, stage=stage)
    ]
    return (
        {'directory': directory.resolve().as_posix(), 'state': state.value, 'instances': instances},
        state,
    )


def _status_document(
    *, entries: list[dict[str, object]], states: Sequence[StudyState]
) -> dict[str, object]:
    return {
        'schema': 'tslab.run-status/1',
        'exit_code': states_exit_code(states=states, live=False),
        'runs': entries,
    }


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
    run_state = shown_run_state(directory=directory, snapshot=snapshot)
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
    log_table(
        logger=_LOGGER, header=('instance', 'state', 'fingerprint', 'time', 'note'), rows=rows
    )
    per_stage: dict[str, Counter[str]] = {}
    for instance_id, record in selected:
        counts = per_stage.setdefault(_stage_of(instance_id=instance_id), Counter())
        counts[_instance_state_shown(record=record, run_state=run_state)] += 1
    for stage_name, counts in per_stage.items():
        summary = ', '.join(f'{count} {state}' for state, count in sorted(counts.items()))
        _LOGGER.info('%s: %s', stage_name, summary)

    _log_run_hints(directory=directory, selected=selected, run_state=run_state)


def _log_run_hints(
    *, directory: Path, selected: list[tuple[str, InstanceRecord]], run_state: str
) -> None:
    failed: list[tuple[str, str]] = []
    for instance_id, record in selected:
        if record.state is InstanceState.FAILED and record.traceback_file is not None:
            error_file = (directory / record.traceback_file).as_posix()
            failed.append((instance_id, error_file))
    for instance_id, error_file in failed[:_MAX_FAILED_INSTANCES]:
        _LOGGER.warning('%s failed; full error: %s', instance_id, error_file)
    if len(failed) > _MAX_FAILED_INSTANCES:
        _LOGGER.warning('… and %d more failed instances', len(failed) - _MAX_FAILED_INSTANCES)
    if run_state == _INTERRUPTED:
        _LOGGER.warning('run interrupted; run the same script again to resume it')
    if selected and all(record.state is InstanceState.DONE for _, record in selected):
        _LOGGER.info('all %d instances done', len(selected))
    if run_state == RunState.RUNNING.value:
        _LOGGER.info(
            'follow active run: %s',
            shlex.join(['tslab', 'status', directory.as_posix(), '--watch', '60']),
        )


def _print_runs(*, root: Path, runs: Sequence[Path], stage: str | None) -> None:
    rows: list[tuple[str, ...]] = []
    totals: Counter[str] = Counter()
    has_failed_instances = False
    for directory in runs:
        try:
            snapshot = read_status(path=directory / STATUS_FILE)
        except StatusFileError as error:
            _LOGGER.warning('skipping an unreadable run: %s', error)
            continue
        if snapshot is None:  # removed while scanning
            continue
        run_state = shown_run_state(directory=directory, snapshot=snapshot)
        totals[run_state] += 1
        # Raw instance states: a dead run still shows its in-flight instances.
        counts = Counter(record.state for _, record in _selected(snapshot=snapshot, stage=stage))
        has_failed_instances = has_failed_instances or counts[InstanceState.FAILED] > 0
        rows.append(
            (
                directory.relative_to(root).as_posix(),
                run_state,
                *(str(counts[state]) for state in _COUNTED_STATES),
            )
        )
    header = ('run', 'state', *(state.value for state in _COUNTED_STATES))
    log_table(logger=_LOGGER, header=header, rows=rows)
    summary = ', '.join(f'{count} {state}' for state, count in sorted(totals.items()))
    _LOGGER.info('%d %s: %s', len(rows), 'run' if len(rows) == 1 else 'runs', summary)
    if totals[RunState.FAILED.value] or totals[_INTERRUPTED] or has_failed_instances:
        _LOGGER.info('details of one run: tslab status <run folder>')
    if totals[RunState.RUNNING.value]:
        _LOGGER.info(
            'follow active runs: %s',
            shlex.join(['tslab', 'status', root.as_posix(), '--watch', '60']),
        )

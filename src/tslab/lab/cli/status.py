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
import os
from pathlib import Path
import shlex
from typing import Annotated

import tyro

from tslab.lab.cli.common import print_json, UsageError, watch
from tslab.lab.core.lock import is_run_active
from tslab.lab.core.paths import STATUS_FILE
from tslab.lab.core.status import (
    InstanceRecord,
    read_status,
    RunStatusSnapshot,
    shown_state_of,
    StatusFileError,
)
from tslab.lab.core.text import log_table
from tslab.lab.enums import InstanceState, JobPhase, RunState, ShownRunState, StudyState
from tslab.lab.study.project import load_project_settings
from tslab.lab.study.reporting.outcome import states_exit_code
from tslab.lab.study.reporting.state import FolderFacts, study_state

__all__ = ['Status']

_LOGGER = logging.getLogger(__name__)
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
        tslab status
        tslab status runs/
        tslab status runs/ --stage scale
        tslab status runs/ --watch 60

    Attributes:
        path: A run folder, or any folder above runs (for example runs/).
            Default: the runs folder of the nearest tslab.toml.
        stage: Count only the instances of this stage.
        watch: Refresh every this many seconds until Ctrl-C.
        json: Print one run-status JSON document.
        exit_code: Return the code for the run states.
    """

    path: Annotated[
        Path | None, tyro.conf.arg(metavar='FOLDER', help_behavior_hint=''), tyro.conf.Positional
    ] = None
    stage: Annotated[str | None, tyro.conf.arg(metavar='STAGE', help_behavior_hint='')] = None
    watch: Annotated[float | None, tyro.conf.arg(metavar='SECONDS', help_behavior_hint='')] = None
    json: bool = False
    exit_code: bool = False

    def run(self) -> int:
        """Print the status.

        Returns:
            ``0`` when something was shown, ``1`` when no run was found or the
            status file of the given run is unreadable, ``2`` when the folder
            does not exist.
        """
        if self.json and self.watch is not None:
            message = '--watch does not combine with --json'
            raise UsageError(message)
        path = self._folder()
        if not path.is_dir():
            _LOGGER.error('%s is not a folder', path.as_posix())
            return 2

        def show() -> int:
            return self._show(path=path)

        return watch(seconds=self.watch, show=show)

    def _folder(self) -> Path:
        """The folder to show: the argument, or the runs folder of tslab.toml."""
        if self.path is not None:
            return self.path
        try:
            settings = load_project_settings(start=Path.cwd())
        except ValueError as error:
            raise UsageError(str(error)) from error
        runs = settings.runs
        source = settings.path
        if runs is None or source is None:
            message = 'give a FOLDER, or set runs in tslab.toml'
            raise UsageError(message)
        _LOGGER.info('runs folder %s from %s', runs.as_posix(), source.as_posix())
        return runs

    def _show(self, *, path: Path) -> int:
        try:
            snapshot = read_status(path=path / STATUS_FILE)
        except StatusFileError as error:
            _LOGGER.error('%s', error)  # noqa: TRY400 # why: a traceback adds nothing
            return 1
        if snapshot is not None:
            return self._show_run(path=path, snapshot=snapshot)
        return self._show_runs(path=path)

    def _show_run(self, *, path: Path, snapshot: RunStatusSnapshot) -> int:
        view = _view_of(directory=path, snapshot=snapshot)
        _print_run(view=view, stage=self.stage)
        if not (self.json or self.exit_code):
            return 0
        entry, state = _run_status_entry(view=view, stage=self.stage)
        code = states_exit_code(states=[state], live=False)
        if self.json:
            print_json(document=_status_document(entries=[entry], states=[state]))
        return code if self.exit_code else 0

    def _show_runs(self, *, path: Path) -> int:
        folders = _run_folders(root=path)
        if not folders:
            _LOGGER.warning('no runs below %s', path.as_posix())
            _LOGGER.info('a run folder holds status.json; pass it, or a folder above runs')
            return 1
        views = _views_of(folders=folders)
        _print_runs(root=path, views=views, stage=self.stage)
        if not (self.json or self.exit_code):
            return 0
        entries: list[dict[str, object]] = []
        states: list[StudyState] = []
        for view in views:
            entry, state = _run_status_entry(view=view, stage=self.stage)
            entries.append(entry)
            states.append(state)
        code = states_exit_code(states=states, live=False)
        if self.json:
            print_json(document=_status_document(entries=entries, states=states))
        return code if self.exit_code else 0


@dataclass(frozen=True, kw_only=True)
class _RunView:
    """One run folder read once: the snapshot, the lock and the state to show.

    Attributes:
        directory: The run folder.
        snapshot: Its ``status.json``.
        folder: The facts of the folder, for the study state.
        run_state: The state to show, ``interrupted`` for a dead ``running`` run.
    """

    directory: Path
    snapshot: RunStatusSnapshot
    folder: FolderFacts
    run_state: ShownRunState


def _view_of(*, directory: Path, snapshot: RunStatusSnapshot) -> _RunView:
    """Read the lock of the folder once and decide the state to show."""
    # why: only a run saved as running can be live; the lock of any other run is not read
    lock_live = snapshot.run_state is RunState.RUNNING and is_run_active(directory=directory)
    return _RunView(
        directory=directory,
        snapshot=snapshot,
        folder=FolderFacts(snapshot=snapshot, read_error=None, lock_live=lock_live),
        run_state=shown_state_of(run_state=snapshot.run_state, lock_live=lock_live),
    )


def _run_folders(*, root: Path) -> list[Path]:
    """The run folders below ``root``, sorted; a folder is not searched below a run."""
    found: list[Path] = []
    for current, names, files in os.walk(root):
        if STATUS_FILE in files:
            found.append(Path(current))
            # why: the folders below a run hold its artifacts, which can be many
            names.clear()
    return sorted(found)


def _views_of(*, folders: Sequence[Path]) -> list[_RunView]:
    """Read every run folder once; an unreadable or vanished one is skipped."""
    views: list[_RunView] = []
    for directory in folders:
        try:
            snapshot = read_status(path=directory / STATUS_FILE)
        except StatusFileError as error:
            _LOGGER.warning('skipping an unreadable run: %s', error)
            continue
        if snapshot is not None:  # None: removed while scanning
            views.append(_view_of(directory=directory, snapshot=snapshot))
    return views


def _run_status_entry(*, view: _RunView, stage: str | None) -> tuple[dict[str, object], StudyState]:
    directory = view.directory
    state, _ = study_state(job_reason=None, job_phase=JobPhase.UNKNOWN, folder=view.folder)
    run_state = view.run_state
    instances = [
        {
            'instance': instance_id,
            'stage': _stage_of(instance_id=instance_id),
            'state': _instance_state_shown(record=record, run_state=run_state),
            'fingerprint': record.fingerprint,
            'seconds': record.seconds,
            'error': record.error,
        }
        for instance_id, record in _selected(snapshot=view.snapshot, stage=stage)
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


def _instance_state_shown(*, record: InstanceRecord, run_state: ShownRunState) -> str:
    if record.state is InstanceState.RUNNING and run_state is ShownRunState.INTERRUPTED:
        return ShownRunState.INTERRUPTED
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


def _print_run(*, view: _RunView, stage: str | None) -> None:
    directory = view.directory
    run_state = view.run_state
    selected = _selected(snapshot=view.snapshot, stage=stage)
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
    *, directory: Path, selected: list[tuple[str, InstanceRecord]], run_state: ShownRunState
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
    if run_state is ShownRunState.INTERRUPTED:
        _LOGGER.warning('run interrupted; run the same script again to resume it')
    if selected and all(record.state is InstanceState.DONE for _, record in selected):
        _LOGGER.info('all %d instances done', len(selected))
    if run_state is ShownRunState.RUNNING:
        _LOGGER.info(
            'follow active run: %s',
            shlex.join(['tslab', 'status', directory.as_posix(), '--watch', '60']),
        )


def _print_runs(*, root: Path, views: Sequence[_RunView], stage: str | None) -> None:
    rows: list[tuple[str, ...]] = []
    totals: Counter[ShownRunState] = Counter()
    has_failed_instances = False
    for view in views:
        directory = view.directory
        run_state = view.run_state
        totals[run_state] += 1
        # Raw instance states: a dead run still shows its in-flight instances.
        counts = Counter(
            record.state for _, record in _selected(snapshot=view.snapshot, stage=stage)
        )
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
    if totals[ShownRunState.FAILED] or totals[ShownRunState.INTERRUPTED] or has_failed_instances:
        _LOGGER.info('details of one run: tslab status <run folder>')
    if totals[ShownRunState.RUNNING]:
        _LOGGER.info(
            'follow active runs: %s',
            shlex.join(['tslab', 'status', root.as_posix(), '--watch', '60']),
        )

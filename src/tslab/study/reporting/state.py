"""The state of each study run, decided from its folder and backend job state."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import time
from typing import TYPE_CHECKING

from tslab.core.lock import is_run_active
from tslab.core.paths import STATUS_FILE
from tslab.core.status import read_status, RunStatusSnapshot
from tslab.core.text import first_line
from tslab.enums import InstanceState, JobFlag, JobPhase, RunState, StudyState
from tslab.study.definition import StudyRun
from tslab.study.records import JobRecord, latest_tasks, TaskRecord

if TYPE_CHECKING:
    from tslab.study.launchers.common.types import JobStates

__all__ = ['FolderFacts', 'RunReport', 'collect_states', 'read_folder', 'study_state']

_INTERRUPTED_ERROR_PREFIXES = ('TrainingPaused:', 'JobTimeLimitSignal:')


@dataclass(frozen=True, kw_only=True)
class FolderFacts:
    """What one run folder reports; a bad ``status.json`` is reported, not raised.

    Args:
        snapshot: The ``status.json`` snapshot, ``None`` when absent or bad.
        read_error: ``"<Type>: <message>"`` when the file was unreadable.
        lock_live: Whether a live process holds the run lock.
    """

    snapshot: RunStatusSnapshot | None
    read_error: str | None
    lock_live: bool


def read_folder(*, run_directory: Path) -> FolderFacts:
    """The facts of one run folder; never raises for a malformed ``status.json``.

    Args:
        run_directory: The run folder.

    Returns:
        The snapshot, read error, and lock state of the folder.
    """
    try:
        snapshot = read_status(path=run_directory / STATUS_FILE)
        read_error = None
    except (TypeError, ValueError, OSError) as error:
        snapshot = None
        read_error = f'{type(error).__name__}: {error}'
    return FolderFacts(
        snapshot=snapshot, read_error=read_error, lock_live=is_run_active(directory=run_directory)
    )


@dataclass(frozen=True, kw_only=True)
class RunReport:
    """The state of one run, for display.

    Args:
        run: The run the report is for.
        state: The decided state.
        reason: A one-line reason, when the state is not obvious.
        job_id: ``<job>_<index>`` for SLURM arrays, the local id otherwise;
            ``None`` when the run was never submitted.
        job_state: The raw state of that task id, when queried.
        job_phase: The backend phase of the task, or ``UNKNOWN`` when absent.
        job_reason: The one-line reason the backend gives for that task id.
        job_flags: What the backend says that state means for the state logic.
        attempt: The attempt of the latest submission; 0 when never submitted.
        task: The latest submitted task of the run.
        folder: The facts of the run folder.
    """

    run: StudyRun
    state: StudyState
    reason: str | None
    job_id: str | None
    job_state: str | None
    job_phase: 'JobPhase'
    attempt: int
    task: TaskRecord | None
    folder: FolderFacts
    job_reason: str | None = None
    job_flags: 'frozenset[JobFlag]' = frozenset()


def _reason(*, folder: FolderFacts, job_reason: str | None) -> str | None:
    snapshot = folder.snapshot
    failed = () if snapshot is None else snapshot.instances.values()
    error = next(
        (
            record.error
            for record in failed
            if record.state is InstanceState.FAILED and record.error is not None
        ),
        None,
    )
    if error is not None:
        return first_line(text=error)
    if folder.read_error is not None:
        return folder.read_error
    return job_reason


def study_state(
    *, job_reason: str | None, job_phase: 'JobPhase', folder: FolderFacts
) -> tuple[StudyState, str | None]:
    """The state of one run, from its folder and the phase of its latest job.

    Args:
        job_reason: The reason the backend gives for the latest task id of the
            run, or ``None`` when the run was never submitted or it gives none.
        job_phase: The backend phase of the latest task id of the run.
        folder: The facts of the run folder.

    Returns:
        The state and a one-line reason, in the order of spec §7.
    """
    snapshot = folder.snapshot
    if job_phase is JobPhase.QUEUED:
        return StudyState.QUEUED, None
    live = folder.lock_live and job_phase is JobPhase.UNKNOWN
    if job_phase is JobPhase.RUNNING or live:
        return StudyState.RUNNING, None
    if (
        snapshot is not None
        and snapshot.run_state is RunState.FINISHED
        and all(record.state is InstanceState.DONE for record in snapshot.instances.values())
    ):
        return StudyState.DONE, None
    # if the lock is live, the function already returned RUNNING
    time_limited = snapshot is not None and any(
        record.state is InstanceState.FAILED
        and (record.error or '').startswith(_INTERRUPTED_ERROR_PREFIXES)
        for record in snapshot.instances.values()
    )
    stale_running = snapshot is not None and snapshot.run_state in (
        RunState.RUNNING,
        RunState.CRASHED,
    )
    # why: a job that ended out of memory or cancelled left a stale ``running`` file;
    # it failed, so the automatic retry must not restart it (spec section 8)
    killed = stale_running and job_phase is JobPhase.FAILED and not time_limited
    interrupted = (stale_running and not killed) or job_phase is JobPhase.INTERRUPTED
    interrupted = interrupted or time_limited
    if interrupted:
        return StudyState.INTERRUPTED, _reason(folder=folder, job_reason=job_reason)
    failed = snapshot is not None and (
        snapshot.run_state is RunState.FAILED
        or (
            snapshot.run_state is RunState.FINISHED
            and any(
                record.state in (InstanceState.FAILED, InstanceState.SKIPPED)
                for record in snapshot.instances.values()
            )
        )
    )
    failed = failed or killed or folder.read_error is not None
    failed = failed or (snapshot is None and job_phase is JobPhase.FAILED)
    if failed:
        return StudyState.FAILED, _reason(folder=folder, job_reason=job_reason)
    return StudyState.MISSING, None


def _pause(*, seconds: float) -> None:
    time.sleep(seconds)


def _sleep(seconds: float) -> None:
    # why: the sleep of collect_states is called with one positional argument
    _pause(seconds=seconds)


def _needs_grace(*, job_phase: 'JobPhase', folder: FolderFacts) -> bool:
    # why: on shared storage the final ``status.json`` of a completed job can reach this
    # node after the SLURM state does
    if job_phase is not JobPhase.COMPLETED:
        return False
    if folder.snapshot is None:
        return folder.read_error is None
    return folder.snapshot.run_state is RunState.RUNNING


def _grace_wait(
    *,
    directories: dict[PurePosixPath, Path],
    completed_phase: 'JobPhase',
    sleep: Callable[[float], None],
) -> dict[PurePosixPath, FolderFacts]:
    """Re-read the listed folders until each settles, at most 15 rounds (30 after a chmod error).

    Returns:
        The latest facts of every re-read folder.
    """
    limit = 15
    pending = dict(directories)
    settled: dict[PurePosixPath, FolderFacts] = {}
    rounds = 0
    while pending and rounds < limit:
        refreshed = [_refresh_failed(directory=directory) for directory in pending.values()]
        if any(refreshed):
            limit = 30
        sleep(1)
        rounds += 1
        settled.update(_reread(pending=pending, completed_phase=completed_phase))
    return settled


def _refresh_failed(*, directory: Path) -> bool:
    """Whether a chmod to the same mode failed; it makes a network file system refresh its cache."""
    try:
        directory.chmod(directory.stat().st_mode)
    except FileNotFoundError:
        # why: a missing folder has no cache to refresh
        return False
    except OSError:
        return True
    return False


def _reread(
    *, pending: dict[PurePosixPath, Path], completed_phase: 'JobPhase'
) -> dict[PurePosixPath, FolderFacts]:
    """Read the pending folders again; the ones that settled leave ``pending``."""
    read = {key: read_folder(run_directory=directory) for key, directory in pending.items()}
    for key, folder in read.items():
        if not _needs_grace(job_phase=completed_phase, folder=folder):
            del pending[key]
    return read


def collect_states(
    *,
    study_directory: Path,
    runs: Sequence[StudyRun],
    jobs: Sequence[JobRecord],
    job_states: 'JobStates',
    sleep: Callable[[float], None] = _sleep,
) -> list[RunReport]:
    """The state report of every run of one study.

    A run whose job completed but whose folder does not say so yet is read again
    for up to 15 seconds (30 if the folder cannot be refreshed), then decided with
    what is there.

    Args:
        study_directory: The ``<output_root>/<study>`` directory.
        runs: The runs of the study, in grid order.
        jobs: The job records of the study, in submission order.
        job_states: The raw states and phases reported by the machine backend.
        sleep: The wait between two reads; injected for the tests.

    Returns:
        One report per run, in the order of ``runs``.
    """
    latest = latest_tasks(jobs=jobs)
    entries = []
    folders: dict[PurePosixPath, FolderFacts] = {}
    waiting: dict[PurePosixPath, Path] = {}
    for run in runs:
        directory = study_directory / run.run_path
        folders[run.run_path] = read_folder(run_directory=directory)
        job, task = latest.get(run.run_path, (None, None))
        job_id = None
        task_id = None
        if job is not None and task is not None:
            task_id = f'{job.job_id}_{task.index}'
            job_id = task_id
        job_state = None if task_id is None else job_states.states.get(task_id)
        job_reason = None if task_id is None else job_states.reasons.get(task_id)
        job_flags = frozenset() if task_id is None else job_states.flags.get(task_id, frozenset())
        job_phase = (
            JobPhase.UNKNOWN
            if task_id is None
            else job_states.phases.get(task_id, JobPhase.UNKNOWN)
        )
        entries.append((run, job_id, job_state, job_reason, job_flags, job_phase, task))
        if _needs_grace(job_phase=job_phase, folder=folders[run.run_path]):
            waiting[run.run_path] = directory
    folders.update(
        _grace_wait(directories=waiting, completed_phase=JobPhase.COMPLETED, sleep=sleep)
    )
    reports: list[RunReport] = []
    for run, job_id, job_state, job_reason, job_flags, job_phase, task in entries:
        folder = folders[run.run_path]
        state, reason = study_state(job_reason=job_reason, job_phase=job_phase, folder=folder)
        reports.append(
            RunReport(
                run=run,
                state=state,
                reason=reason,
                job_id=job_id,
                job_state=job_state,
                job_phase=job_phase,
                job_reason=job_reason,
                job_flags=job_flags,
                attempt=task.attempt if task is not None else 0,
                task=task,
                folder=folder,
            )
        )
    return reports

"""The state of each study run, decided from its folder and its SLURM job."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
import time

from tslab.core.lock import is_run_active
from tslab.core.paths import STATUS_FILE
from tslab.core.status import InstanceState, read_status, RunState, RunStatusSnapshot
from tslab.core.text import first_line
from tslab.jobs.slurm import (
    FAILING_JOB_STATES,
    INTERRUPTING_JOB_STATES,
    job_ended,
    QUEUED_JOB_STATES,
    RUNNING_JOB_STATES,
)
from tslab.study.definition import StudyRun
from tslab.study.records import JobRecord, latest_tasks, TaskRecord

__all__ = [
    'LOCAL_ENDED',
    'FolderFacts',
    'RunReport',
    'StudyState',
    'collect_states',
    'followup_job_ids',
    'read_folder',
    'slurm_task_ids',
    'study_state',
]

#: The job state of a local task that is not in a live launch.
LOCAL_ENDED = 'ENDED'
_INTERRUPTED_ERROR_PREFIXES = ('TrainingPaused:', 'JobTimeLimitSignal:')


class StudyState(StrEnum):
    """The state shown for one run of a study."""

    QUEUED = 'queued'
    RUNNING = 'running'
    DONE = 'done'
    INTERRUPTED = 'interrupted'
    FAILED = 'failed'
    MISSING = 'missing'


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
        job_state: The SLURM state of that task id, when queried.
        attempt: The attempt of the latest submission; 0 when never submitted.
        task: The latest submitted task of the run.
        folder: The facts of the run folder.
    """

    run: StudyRun
    state: StudyState
    reason: str | None
    job_id: str | None
    job_state: str | None
    attempt: int
    task: TaskRecord | None
    folder: FolderFacts


def _reason(*, folder: FolderFacts, job_state: str | None) -> str | None:
    snapshot = folder.snapshot
    if snapshot is not None:
        for record in snapshot.instances.values():
            if record.state is InstanceState.FAILED and record.error is not None:
                return first_line(text=record.error)
    if folder.read_error is not None:
        return folder.read_error
    if job_state is not None and job_state != LOCAL_ENDED:
        return f'SLURM {job_state}'
    return None


def study_state(*, job_state: str | None, folder: FolderFacts) -> tuple[StudyState, str | None]:
    """The state of one run, from its folder and the state of its SLURM task.

    Args:
        job_state: The SLURM state of the latest task id of the run, or ``None``
            when the run was never submitted or SLURM is unavailable.
        folder: The facts of the run folder.

    Returns:
        The state and a one-line reason, in the order of spec §7.
    """
    snapshot = folder.snapshot
    if job_state in QUEUED_JOB_STATES:
        return StudyState.QUEUED, None
    live = folder.lock_live and (job_state is None or not job_ended(job_state=job_state))
    if job_state in RUNNING_JOB_STATES or live:
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
    killed = stale_running and job_state in FAILING_JOB_STATES and not time_limited
    interrupted = (stale_running and not killed) or job_state in INTERRUPTING_JOB_STATES
    interrupted = interrupted or time_limited
    if interrupted:
        return StudyState.INTERRUPTED, _reason(folder=folder, job_state=job_state)
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
    failed = failed or (snapshot is None and job_state in FAILING_JOB_STATES)
    if failed:
        return StudyState.FAILED, _reason(folder=folder, job_state=job_state)
    return StudyState.MISSING, None


def slurm_task_ids(*, jobs: Sequence[JobRecord]) -> list[str]:
    """The task ids to query: the latest SLURM task of every run.

    Args:
        jobs: The job records of the study, in submission order.

    Returns:
        The ``<job>_<index>`` ids of the latest slurm-submitted tasks,
        sorted; local jobs hold no task ids.
    """
    ids = {
        f'{job.job_id}_{task.index}'
        for job, task in latest_tasks(jobs=jobs).values()
        if job.launcher == 'slurm'
    }
    return sorted(ids)


def followup_job_ids(*, jobs: Sequence[JobRecord]) -> list[str]:
    """The sorted SLURM ids of follow-up jobs.

    Args:
        jobs: The job records of the study, in submission order.

    Returns:
        The ids of SLURM jobs recorded as follow-ups.
    """
    return sorted(
        {job.job_id for job in jobs if job.kind == 'followup' and job.launcher == 'slurm'}
    )


def _pause(seconds: float) -> None:
    time.sleep(seconds)


def _needs_grace(*, job_state: str | None, folder: FolderFacts) -> bool:
    # why: on shared storage the final ``status.json`` of a completed job can reach this
    # node after the SLURM state does
    if job_state != 'COMPLETED':
        return False
    if folder.snapshot is None:
        return folder.read_error is None
    return folder.snapshot.run_state is RunState.RUNNING


def _grace_wait(
    *, directories: dict[str, Path], facts: dict[str, FolderFacts], sleep: Callable[[float], None]
) -> None:
    """Re-read the listed folders until each settles, at most 15 rounds (30 after a chmod error)."""
    limit = 15
    pending = dict(directories)
    rounds = 0
    while pending and rounds < limit:
        for directory in pending.values():
            try:
                # why: a chmod to the same mode makes a network file system refresh its cache
                directory.chmod(directory.stat().st_mode)
            except OSError:
                limit = 30
        sleep(1)
        rounds += 1
        for key, directory in list(pending.items()):
            facts[key] = read_folder(run_directory=directory)
            if not _needs_grace(job_state='COMPLETED', folder=facts[key]):
                del pending[key]


def collect_states(
    *,
    study_directory: Path,
    runs: Sequence[StudyRun],
    jobs: Sequence[JobRecord],
    job_states: Mapping[str, str] | None,
    sleep: Callable[[float], None] = _pause,
) -> list[RunReport]:
    """The state report of every run of one study.

    A run whose job completed but whose folder does not say so yet is read again
    for up to 15 seconds (30 if the folder cannot be refreshed), then decided with
    what is there.

    Args:
        study_directory: The ``<output_root>/<study>`` directory.
        runs: The runs of the study, in grid order.
        jobs: The job records of the study, in submission order.
        job_states: The SLURM states per task id, or ``None`` when SLURM is
            unavailable.
        sleep: The wait between two reads; injected for the tests.

    Returns:
        One report per run, in the order of ``runs``.
    """
    latest = latest_tasks(jobs=jobs)
    entries = []
    folders: dict[str, FolderFacts] = {}
    waiting: dict[str, Path] = {}
    for run in runs:
        directory = study_directory / run.run_path
        folders[run.run_path] = read_folder(run_directory=directory)
        job, task = latest.get(run.run_path, (None, None))
        job_id = None
        task_id = None
        if job is not None and task is not None:
            task_id = f'{job.job_id}_{task.index}'
            job_id = task_id
        job_state = None if job_states is None or task_id is None else job_states.get(task_id)
        entries.append((run, job_id, job_state, task))
        if _needs_grace(job_state=job_state, folder=folders[run.run_path]):
            waiting[run.run_path] = directory
    _grace_wait(directories=waiting, facts=folders, sleep=sleep)
    reports: list[RunReport] = []
    for run, job_id, job_state, task in entries:
        folder = folders[run.run_path]
        state, reason = study_state(job_state=job_state, folder=folder)
        reports.append(
            RunReport(
                run=run,
                state=state,
                reason=reason,
                job_id=job_id,
                job_state=job_state,
                attempt=task.attempt if task is not None else 0,
                task=task,
                folder=folder,
            )
        )
    return reports

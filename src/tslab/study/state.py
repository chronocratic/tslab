"""The state of each study run, decided from its folder and its SLURM job."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from tslab.core.run import is_run_active
from tslab.core.status import read_status, RunState, RunStatusSnapshot
from tslab.study.definition import StudyRun
from tslab.study.records import JobRecord, latest_tasks, TaskRecord

__all__ = [
    'FAILING_JOB_STATES',
    'INTERRUPTING_JOB_STATES',
    'QUEUED_JOB_STATES',
    'RUNNING_JOB_STATES',
    'FolderFacts',
    'RunReport',
    'StudyState',
    'collect_states',
    'read_folder',
    'slurm_task_ids',
    'study_state',
]

#: SLURM states in which the job has not started yet.
QUEUED_JOB_STATES = frozenset({'PENDING', 'REQUEUED'})

#: SLURM states in which the job is actively working.
RUNNING_JOB_STATES = frozenset(
    {'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED', 'STAGE_OUT', 'SIGNALING', 'RESIZING'}
)

#: Ended SLURM states that read as an interrupted run, not a failed one.
INTERRUPTING_JOB_STATES = frozenset({'TIMEOUT', 'NODE_FAIL', 'PREEMPTED'})

#: Ended SLURM states that read as a failed run.
FAILING_JOB_STATES = frozenset({'OUT_OF_MEMORY', 'FAILED', 'CANCELLED', 'BOOT_FAIL', 'DEADLINE'})


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
        snapshot = read_status(path=run_directory / 'status.json')
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


def _ended(job_state: str | None) -> bool:
    return (
        job_state is not None
        and job_state not in QUEUED_JOB_STATES
        and job_state not in RUNNING_JOB_STATES
    )


def _reason(*, folder: FolderFacts, job_state: str | None) -> str | None:
    snapshot = folder.snapshot
    if snapshot is not None:
        for record in snapshot.instances.values():
            if record.state.value == 'failed' and record.error is not None:
                return record.error.splitlines()[0]
    if folder.read_error is not None:
        return folder.read_error
    if job_state is not None:
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
    live = folder.lock_live and (job_state is None or not _ended(job_state))
    if job_state in RUNNING_JOB_STATES or live:
        return StudyState.RUNNING, None
    if (
        snapshot is not None
        and snapshot.run_state is RunState.FINISHED
        and all(record.state.value == 'done' for record in snapshot.instances.values())
    ):
        return StudyState.DONE, None
    # if the lock is live, the function already returned RUNNING
    interrupted = snapshot is not None and snapshot.run_state in (
        RunState.RUNNING,
        RunState.CRASHED,
    )
    interrupted = interrupted or job_state in INTERRUPTING_JOB_STATES
    interrupted = interrupted or (
        snapshot is not None
        and any(
            record.state.value == 'failed'
            and (record.error or '').startswith('TrainingTimeLimitReached:')
            for record in snapshot.instances.values()
        )
    )
    if interrupted:
        return StudyState.INTERRUPTED, _reason(folder=folder, job_state=job_state)
    failed = snapshot is not None and (
        snapshot.run_state is RunState.FAILED
        or (
            snapshot.run_state is RunState.FINISHED
            and any(
                record.state.value in ('failed', 'skipped')
                for record in snapshot.instances.values()
            )
        )
    )
    failed = failed or folder.read_error is not None
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


def collect_states(
    *,
    study_directory: Path,
    runs: Sequence[StudyRun],
    jobs: Sequence[JobRecord],
    job_states: Mapping[str, str] | None,
) -> list[RunReport]:
    """The state report of every run of one study.

    Args:
        study_directory: The ``<output_root>/<study>`` directory.
        runs: The runs of the study, in grid order.
        jobs: The job records of the study, in submission order.
        job_states: The SLURM states per task id, or ``None`` when SLURM is
            unavailable.

    Returns:
        One report per run, in the order of ``runs``.
    """
    latest = latest_tasks(jobs=jobs)
    reports: list[RunReport] = []
    for run in runs:
        folder = read_folder(run_directory=study_directory / run.run_path)
        job, task = latest.get(run.run_path, (None, None))
        job_id = None
        task_id = None
        if job is not None and task is not None:
            task_id = f'{job.job_id}_{task.index}' if job.launcher == 'slurm' else job.job_id
            job_id = task_id
        job_state = None if job_states is None or task_id is None else job_states.get(task_id)
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

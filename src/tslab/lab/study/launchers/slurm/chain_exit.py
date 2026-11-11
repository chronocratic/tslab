"""Exit status for chained SLURM tasks after their run command completes."""

from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
import dataclasses
import logging
import os
from pathlib import Path
import time
from types import MappingProxyType

from tslab.lab.core.status import utc_now_iso
from tslab.lab.enums import ChainedRetry, JobPhase, StudyState
from tslab.lab.jobs.slurm import query_cluster_config, RESTART_COUNT_VARIABLE, SlurmError
from tslab.lab.study.commands.resubmit import may_retry
from tslab.lab.study.context import Context
from tslab.lab.study.definition.definition import expand_runs
from tslab.lab.study.launchers.common.types import JobStates
from tslab.lab.study.launchers.slurm.requeue import (
    at_requeue_limit,
    requeue_task,
    restarted_attempt,
)
from tslab.lab.study.paths import study_folder
from tslab.lab.study.records import (
    append_job,
    JobRecord,
    ProgressSnapshot,
    read_jobs,
    submission_lock,
    SubmissionLockTimeoutError,
    take_progress,
    TaskRecord,
)
from tslab.lab.study.reporting.state import collect_states, RunReport

__all__ = ['chain_exit']

_LOGGER = logging.getLogger(__name__)

# The SLURM time-limit signal comes shortly before the kill; waiting longer for the
# submission lock would lose the requeue anyway.
_LOCK_WAIT_SECONDS = 30


def chain_exit(*, context: Context, job_file: int, index: int) -> int:
    """Return success only when the recorded task run finished successfully.

    Args:
        context: The context of the study whose task completed.
        job_file: The ``jobs/<n>`` file number of the submitted task.
        index: The array index of the task.

    Returns:
        ``0`` when the run is ``DONE`` or was requeued in place; otherwise ``1``. The
        job script still exits with a non-zero exit code of the run command first, so
        dependent jobs see a failed task even when this returns ``0``.
    """
    jobs_path = study_folder(machine=context.machine, study=context.study) / 'submissions.jsonl'
    loaded = _load_report(
        context=context, jobs=context.jobs, job_file=job_file, index=index, sleep=time.sleep
    )
    if loaded is None:
        loaded = _load_report(
            context=context,
            jobs=read_jobs(path=jobs_path),
            job_file=job_file,
            index=index,
            sleep=time.sleep,
        )
    if loaded is None:
        return 1
    report = loaded.report
    if report.state is StudyState.DONE:
        return 0
    if report.state is StudyState.INTERRUPTED and _requeued(
        context=context,
        loaded=loaded,
        job_file=job_file,
        index=index,
        jobs_path=jobs_path,
        environment=os.environ,
    ):
        return 0
    return 1


@dataclasses.dataclass(frozen=True, kw_only=True)
class _TaskReport:
    """The records of one task and its report after the task command ended.

    Attributes:
        matches: Every record and task of the job file and index, the submission first.
        report: The report of the run, with the task counted as completed.
    """

    matches: list[tuple[JobRecord, TaskRecord]]
    report: RunReport


def _no_wait(_seconds: float) -> None:
    """Skip the wait between two reads of a run folder."""


def _load_report(
    *,
    context: Context,
    jobs: Sequence[JobRecord],
    job_file: int,
    index: int,
    sleep: Callable[[float], None],
) -> _TaskReport | None:
    """The report of one task from the given records, or ``None`` when it is unknown."""
    matches = _job_tasks_of(jobs=jobs, job_file=job_file, index=index)
    if not matches:
        return None
    job, task = matches[0]
    run = next(
        (run for run in expand_runs(study=context.study) if run.run_path == task.run_path), None
    )
    if run is None:
        return None
    task_id = f'{job.job_id}_{task.index}'
    reports = collect_states(
        study_directory=context.study_directory,
        runs=[run],
        jobs=[record for record, _ in matches],
        job_states=JobStates(
            known=True,
            states=MappingProxyType({}),
            phases=MappingProxyType({task_id: JobPhase.COMPLETED}),
            reasons=MappingProxyType({}),
            flags=MappingProxyType({}),
            followup_live_job_ids=(),
        ),
        sleep=sleep,
    )
    return _TaskReport(matches=matches, report=reports[0])


def _requeued(  # noqa: PLR0913 # why: keyword-only private helper, one per fact of the requeue
    *,
    context: Context,
    loaded: _TaskReport,
    job_file: int,
    index: int,
    jobs_path: Path,
    environment: Mapping[str, str],
) -> bool:
    """Requeue the task in place when the mode and the retry rules allow it.

    Each requeue by TSLab appends a record of the job with the next attempt and a
    fresh progress snapshot, so ``status``, the progress rule and the follow-up see
    it. Restarts beyond those records were requeues by SLURM itself (preemption,
    node failure) and count as uncounted attempts.

    The lock is held from reading the records to writing the next attempt, so a
    concurrent resubmit never creates the same attempt twice. A busy lock is
    waited for at most ``_LOCK_WAIT_SECONDS``: this process is inside the window
    between the time-limit signal and the kill, so a long wait would lose the
    requeue anyway; the follow-up resubmits the run instead.
    """
    if context.machine.followup.chained_retry is not ChainedRetry.REQUEUE:
        return False
    # why: the checkpoint search walks the run folder once, before the lock is held; the
    # scheduler query can take 30 s, so it runs only when a retry is possible
    restart_count = int(environment.get(RESTART_COUNT_VARIABLE, '0'))
    progress = take_progress(run_directory=context.study_directory / loaded.report.run.run_path)
    retry = _retryable_task(
        context=context, loaded=loaded, restart_count=restart_count, progress=progress
    )
    if retry is None:
        return False
    if at_requeue_limit(restart_count=restart_count, max_requeues=_max_batch_requeue()):
        _LOGGER.warning(
            '%s is at MaxBatchRequeue; ending the job so the follow-up resubmits it', retry.job_id
        )
        return False
    with ExitStack() as stack:
        try:
            stack.enter_context(
                submission_lock(study_folder=jobs_path.parent, timeout_seconds=_LOCK_WAIT_SECONDS)
            )
        except SubmissionLockTimeoutError:
            _LOGGER.warning(
                'the submission lock stayed busy for %d s; skipping the requeue so the '
                'follow-up resubmits the task',
                _LOCK_WAIT_SECONDS,
            )
            return False
        return _requeue_locked(
            context=context,
            job_file=job_file,
            index=index,
            jobs_path=jobs_path,
            restart_count=restart_count,
            progress=progress,
        )


@dataclasses.dataclass(frozen=True, kw_only=True)
class _Retry:
    """The task of an interrupted attempt that may be requeued.

    Attributes:
        job_id: The scheduler id of the task, e.g. ``41_17``.
        latest: The latest recorded task.
        current: The latest task with its attempts counted.
    """

    job_id: str
    latest: TaskRecord
    current: TaskRecord


def _kept_requeues(*, matches: Sequence[tuple[JobRecord, TaskRecord]]) -> int:
    """Count the requeue records that were not taken back after a failed requeue call."""
    attempts = [task.attempt for _, task in matches]
    return sum(
        1
        for position in range(1, len(attempts))
        if attempts[position] > attempts[position - 1]
        and not (position + 1 < len(attempts) and attempts[position + 1] < attempts[position])
    )


def _retryable_task(
    *, context: Context, loaded: _TaskReport, restart_count: int, progress: ProgressSnapshot
) -> _Retry | None:
    """The task of the interrupted attempt with its attempts counted, or ``None``.

    ``None`` means the task is unknown or ``may_retry`` refuses another attempt.
    """
    report = loaded.report
    _, submitted = loaded.matches[0]
    latest = report.task
    if latest is None or report.job_id is None:
        return None
    restarts = restarted_attempt(attempt=0, restart_count=restart_count)
    current = dataclasses.replace(
        latest,
        attempt=restarted_attempt(attempt=submitted.attempt, restart_count=restart_count),
        uncounted_attempts=submitted.uncounted_attempts
        + max(restarts - _kept_requeues(matches=loaded.matches), 0),
    )
    counted = dataclasses.replace(report, attempt=current.attempt, task=current)
    if not may_retry(report=counted, context=context, progress=progress):
        return None
    return _Retry(job_id=report.job_id, latest=latest, current=current)


def _requeue_locked(  # noqa: PLR0913 # why: keyword-only private helper, one per fact of the requeue
    *,
    context: Context,
    job_file: int,
    index: int,
    jobs_path: Path,
    restart_count: int,
    progress: ProgressSnapshot,
) -> bool:
    """Read the records again, decide and write the next attempt; the lock is held."""
    # why: the grace wait for the run folder already ran before the lock; waiting again
    # here could outlast the window between the time-limit signal and the kill
    loaded = _load_report(
        context=context,
        jobs=read_jobs(path=jobs_path),
        job_file=job_file,
        index=index,
        sleep=_no_wait,
    )
    if loaded is None or loaded.report.state is not StudyState.INTERRUPTED:
        return False
    job, _ = loaded.matches[0]
    retry = _retryable_task(
        context=context, loaded=loaded, restart_count=restart_count, progress=progress
    )
    if retry is None:
        return False
    following = dataclasses.replace(
        retry.current, attempt=retry.current.attempt + 1, progress=progress, uncounted_in_row=0
    )
    # why: the scheduler ends this process soon after the requeue call, so the record
    # of the next attempt is written first and taken back when the call fails
    append_job(
        path=jobs_path, record=dataclasses.replace(job, time=utc_now_iso(), tasks=(following,))
    )
    if not requeue_task(task_id=retry.job_id):
        append_job(
            path=jobs_path,
            record=dataclasses.replace(job, time=utc_now_iso(), tasks=(retry.latest,)),
        )
        return False
    _LOGGER.info('requeued %s (attempt %d)', retry.job_id, following.attempt)
    return True


def _max_batch_requeue() -> int | None:
    """Read and parse the scheduler requeue limit, logging failures at the launcher edge."""
    try:
        value = query_cluster_config().get('MaxBatchRequeue', '')
    except SlurmError as error:
        _LOGGER.warning('could not read MaxBatchRequeue from SLURM: %s', error)
        return None
    try:
        return int(value)
    except ValueError as error:
        _LOGGER.warning('could not parse MaxBatchRequeue from SLURM: %s', error)
        return None


def _job_tasks_of(
    *, jobs: Sequence[JobRecord], job_file: int, index: int
) -> list[tuple[JobRecord, TaskRecord]]:
    """Every record and task of one job file and array index, the submission first."""
    return [
        (job, task)
        for job in jobs
        if job.job_file == job_file
        for task in job.tasks
        if task.index == index
    ]

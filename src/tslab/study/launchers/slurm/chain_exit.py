"""Exit status for chained SLURM tasks after their run command completes."""

from collections.abc import Sequence
import dataclasses
import logging
from pathlib import Path
from types import MappingProxyType

from tslab.core.status import utc_now_iso
from tslab.enums import ChainedRetry, JobPhase, StudyState
from tslab.study.commands.resubmit import may_retry
from tslab.study.context import Context
from tslab.study.definition import expand_runs
from tslab.study.launchers.common.types import JobStates
from tslab.study.launchers.slurm.requeue import requeue_task, restarted_attempt
from tslab.study.paths import study_folder
from tslab.study.records import append_job, JobRecord, read_jobs, take_progress, TaskRecord
from tslab.study.reporting.state import collect_states, RunReport

__all__ = ['chain_exit']

_LOGGER = logging.getLogger(__name__)


def chain_exit(*, context: Context, job_file: int, index: int) -> int:
    """Return success only when the recorded task run finished successfully.

    Args:
        context: The context of the study whose task completed.
        job_file: The ``jobs/<n>`` file number of the submitted task.
        index: The array index of the task.

    Returns:
        ``0`` when the run is ``DONE`` or was requeued in place; otherwise ``1``.
    """
    jobs_path = study_folder(machine=context.machine, study=context.study) / 'submissions.jsonl'
    matches = _job_tasks_of(jobs=context.jobs, job_file=job_file, index=index)
    if not matches:
        matches = _job_tasks_of(jobs=read_jobs(path=jobs_path), job_file=job_file, index=index)
    if not matches:
        return 1
    job, task = matches[0]
    run = next(
        (run for run in expand_runs(study=context.study) if run.run_path == task.run_path), None
    )
    if run is None:
        return 1
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
    )
    report = reports[0]
    if report.state is StudyState.DONE:
        return 0
    if report.state is StudyState.INTERRUPTED and _requeued(
        context=context,
        report=report,
        job=job,
        submitted=task,
        requeues=len(matches) - 1,
        jobs_path=jobs_path,
    ):
        return 0
    return 1


def _requeued(  # noqa: PLR0913 # why: keyword-only private helper, one per recorded fact
    *,
    context: Context,
    report: RunReport,
    job: JobRecord,
    submitted: TaskRecord,
    requeues: int,
    jobs_path: Path,
) -> bool:
    """Requeue the task in place when the mode and the retry rules allow it.

    Each requeue by TSLab appends a record of the job with the next attempt and a
    fresh progress snapshot, so ``status``, the progress rule and the follow-up see
    it. Restarts beyond those records were requeues by SLURM itself (preemption,
    node failure) and count as uncounted attempts.
    """
    if context.machine.followup.chained_retry is not ChainedRetry.REQUEUE:
        return False
    latest = report.task
    if latest is None or report.job_id is None:
        return False
    restarts = restarted_attempt(attempt=0)
    current = dataclasses.replace(
        latest,
        attempt=restarted_attempt(attempt=submitted.attempt),
        uncounted_attempts=submitted.uncounted_attempts + max(restarts - requeues, 0),
    )
    counted = dataclasses.replace(report, attempt=current.attempt, task=current)
    if not may_retry(report=counted, context=context):
        return False
    following = dataclasses.replace(
        current,
        attempt=current.attempt + 1,
        progress=take_progress(run_directory=context.study_directory / report.run.run_path),
        uncounted_in_row=0,
    )
    # why: the scheduler ends this process soon after the requeue call, so the record
    # of the next attempt is written first and taken back when the call fails
    append_job(
        path=jobs_path, record=dataclasses.replace(job, time=utc_now_iso(), tasks=(following,))
    )
    if not requeue_task(task_id=report.job_id):
        append_job(
            path=jobs_path, record=dataclasses.replace(job, time=utc_now_iso(), tasks=(latest,))
        )
        return False
    _LOGGER.info('requeued %s (attempt %d)', report.job_id, following.attempt)
    return True


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

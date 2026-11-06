"""Exit status for chained SLURM tasks after their run command completes."""

from collections.abc import Sequence
import dataclasses
import logging
from types import MappingProxyType

from tslab.enums import ChainedRetry, JobPhase, StudyState
from tslab.study.commands.resubmit import may_retry
from tslab.study.context import Context
from tslab.study.definition import expand_runs
from tslab.study.launchers.common.types import JobStates
from tslab.study.launchers.slurm.requeue import requeue_task, restarted_attempt
from tslab.study.paths import study_folder
from tslab.study.records import JobRecord, read_jobs, TaskRecord
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
        ``0`` when the run is ``DONE``; otherwise ``1``.
    """
    match = _job_task_of(jobs=context.jobs, job_file=job_file, index=index)
    if match is None:
        jobs_path = study_folder(machine=context.machine, study=context.study) / 'submissions.jsonl'
        match = _job_task_of(jobs=read_jobs(path=jobs_path), job_file=job_file, index=index)
    if match is None:
        return 1
    job, task = match
    run = next(
        (run for run in expand_runs(study=context.study) if run.run_path == task.run_path), None
    )
    if run is None:
        return 1
    task_id = f'{job.job_id}_{task.index}'
    reports = collect_states(
        study_directory=context.study_directory,
        runs=[run],
        jobs=[job],
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
        context=context, report=report, task_id=task_id
    ):
        return 0
    return 1


def _requeued(*, context: Context, report: RunReport, task_id: str) -> bool:
    """Requeue the task in place when the mode and the retry rules allow it."""
    if context.machine.followup.chained_retry is not ChainedRetry.REQUEUE:
        return False
    counted = dataclasses.replace(report, attempt=restarted_attempt(attempt=report.attempt))
    if not may_retry(report=counted, context=context):
        return False
    if not requeue_task(task_id=task_id):
        return False
    _LOGGER.info('requeued %s (attempt %d)', task_id, counted.attempt)
    return True


def _job_task_of(
    *, jobs: Sequence[JobRecord], job_file: int, index: int
) -> tuple[JobRecord, TaskRecord] | None:
    """Find one record and task by job file and array index."""
    return next(
        (
            (job, task)
            for job in jobs
            if job.job_file == job_file
            for task in job.tasks
            if task.index == index
        ),
        None,
    )

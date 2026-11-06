"""Exit status for chained SLURM tasks after their run command completes."""

from collections.abc import Sequence

from tslab.enums import JobPhase, StudyState
from tslab.study.context import Context
from tslab.study.paths import study_folder
from tslab.study.records import JobRecord, read_jobs, TaskRecord
from tslab.study.reporting.state import read_folder, study_state

__all__ = ['chain_exit']


def chain_exit(*, context: Context, job_file: int, index: int) -> int:
    """Return success only when the recorded task run finished successfully.

    Args:
        context: The context of the study whose task completed.
        job_file: The ``jobs/<n>`` file number of the submitted task.
        index: The array index of the task.

    Returns:
        ``0`` when the run is ``DONE``; otherwise ``1``.
    """
    task = _task_of(jobs=context.jobs, job_file=job_file, index=index)
    if task is None:
        jobs_path = study_folder(machine=context.machine, study=context.study) / 'submissions.jsonl'
        task = _task_of(jobs=read_jobs(path=jobs_path), job_file=job_file, index=index)
    if task is None:
        return 1
    folder = read_folder(run_directory=context.study_directory / task.run_path)
    state, _ = study_state(job_reason=None, job_phase=JobPhase.COMPLETED, folder=folder)
    return 0 if state is StudyState.DONE else 1


def _task_of(*, jobs: Sequence[JobRecord], job_file: int, index: int) -> TaskRecord | None:
    """Find one task by its job file and array index."""
    return next(
        (
            task
            for job in jobs
            if job.job_file == job_file
            for task in job.tasks
            if task.index == index
        ),
        None,
    )

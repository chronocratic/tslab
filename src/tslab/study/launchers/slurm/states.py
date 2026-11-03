"""Reading SLURM task states and expected start times."""

from collections.abc import Mapping, Sequence
import logging

from tslab.jobs.slurm import (
    job_ended,
    pending_reason_stuck,
    query_expected_starts,
    query_job_states,
    SlurmError,
)
from tslab.study.context import Context
from tslab.study.launchers import QueuedJob
from tslab.study.records import JobKind, JobRecord, latest_tasks
from tslab.study.reporting.state import StudyState

__all__ = ['expected_starts', 'followup_job_ids', 'slurm_task_ids']

_LOGGER = logging.getLogger(__name__)


def slurm_task_ids(*, jobs: Sequence[JobRecord]) -> list[str]:
    """The sorted task ids of the latest numeric SLURM job for every run."""
    ids = {
        f'{job.job_id}_{task.index}'
        for job, task in latest_tasks(jobs=jobs).values()
        if job.job_id.isdecimal()
    }
    return sorted(ids)


def followup_job_ids(*, jobs: Sequence[JobRecord]) -> list[str]:
    """The sorted numeric ids of recorded follow-up jobs."""
    return sorted(
        {job.job_id for job in jobs if job.kind == JobKind.FOLLOWUP and job.job_id.isdecimal()}
    )


def _query_states(
    *, job_ids: list[str], ended_states: Mapping[str, str] | None
) -> tuple[dict[str, str], dict[str, str]]:
    if ended_states is None:
        states = query_job_states(job_ids=job_ids)
        return states, {}
    queried = query_job_states(job_ids=[item for item in job_ids if item not in ended_states])
    # why: a failed query raised above, so the cache only ever holds states SLURM reported
    new_cache = {
        **ended_states,
        **{item: state for item, state in queried.items() if job_ended(job_state=state)},
    }
    return {**ended_states, **queried}, new_cache


def expected_starts(*, context: Context) -> dict[str, QueuedJob] | None:
    """The start time and pending reason of each queued run, or ``None`` when none are queued."""
    queued = [
        report.job_id
        for report in context.reports
        if report.state is StudyState.QUEUED and report.job_id is not None
    ]
    if not queued:
        return None
    try:
        entries = query_expected_starts(job_ids=queued)
    except SlurmError as error:
        _LOGGER.warning('%s', error)
        return None
    return {
        task_id: QueuedJob(
            start=entry.start, reason=entry.reason, stuck=pending_reason_stuck(reason=entry.reason)
        )
        for task_id, entry in entries.items()
    }

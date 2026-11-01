"""Reading SLURM task states and expected start times."""

from collections.abc import Mapping
import logging

from tslab.jobs.slurm import job_ended, query_expected_starts, query_job_states, SlurmError
from tslab.study.context import Context
from tslab.study.state import StudyState

__all__ = ['expected_starts']

_LOGGER = logging.getLogger(__name__)


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


def expected_starts(*, context: Context) -> dict[str, str | None] | None:
    """The start time of each queued run, or ``None`` when none are queued."""
    queued = [
        report.job_id
        for report in context.reports
        if report.state is StudyState.QUEUED and report.job_id is not None
    ]
    if not queued:
        return None
    try:
        return query_expected_starts(job_ids=queued)
    except SlurmError as error:
        _LOGGER.warning('%s', error)
        return None

"""Accounting usage from SLURM."""

from collections.abc import Mapping, Sequence
import logging

from tslab.jobs.slurm import query_job_usage, SlurmError, SlurmJobUsage
from tslab.study.context import Context
from tslab.study.launchers.common.types import JobUsage
from tslab.study.reporting.state import RunReport

__all__ = ['failure_usage', 'usage']

_LOGGER = logging.getLogger(__name__)


def _job_usages(*, usage: Mapping[str, SlurmJobUsage]) -> dict[str, JobUsage]:
    return {
        job_id: JobUsage(
            state=item.state,
            elapsed=item.elapsed,
            time_limit=item.time_limit,
            total_cpu=item.total_cpu,
            max_rss=item.max_rss,
            alloc_cpus=item.alloc_cpus,
            gpus=item.gpus,
            exit_code=item.exit_code,
        )
        for job_id, item in usage.items()
    }


def usage(*, context: Context) -> Mapping[str, JobUsage]:
    """The accounting usage of every submitted run."""
    job_ids = [report.job_id for report in context.reports if report.job_id is not None]
    if not job_ids:
        return dict[str, JobUsage]()
    return _job_usages(usage=query_job_usage(job_ids=job_ids))


def failure_usage(
    *,
    context: Context,  # noqa: ARG001 # why: the Backend method signature
    reports: Sequence[RunReport],
) -> Mapping[str, JobUsage]:
    """The accounting usage of the failed and interrupted runs; no numbers on failure."""
    # why: the run state is the study verdict, so a scheduler state of COMPLETED
    # or a missing state does not keep the recorded exit code away
    job_ids = [report.job_id for report in reports if report.job_id is not None]
    if not job_ids:
        return dict[str, JobUsage]()
    try:
        return _job_usages(usage=query_job_usage(job_ids=job_ids))
    except SlurmError as error:
        _LOGGER.warning('%s', error)
        return dict[str, JobUsage]()

"""Accounting usage from SLURM."""

from collections.abc import Mapping, Sequence
import logging

from tslab.jobs.slurm import query_job_usage, SlurmError, SlurmJobUsage
from tslab.study.context import Context
from tslab.study.state import RunReport

__all__ = ['failure_usage', 'usage']

_LOGGER = logging.getLogger(__name__)


def usage(*, context: Context) -> Mapping[str, SlurmJobUsage]:
    """The accounting usage of every submitted run."""
    job_ids = [report.job_id for report in context.reports if report.job_id is not None]
    if not job_ids:
        return dict[str, SlurmJobUsage]()
    return query_job_usage(job_ids=job_ids)


def failure_usage(
    *,
    context: Context,  # noqa: ARG001 # why: the Backend method signature
    reports: Sequence[RunReport],
) -> Mapping[str, SlurmJobUsage]:
    """The accounting usage of selected out-of-memory runs; no numbers on failure."""
    job_ids = [
        report.job_id
        for report in reports
        if report.job_state == 'OUT_OF_MEMORY' and report.job_id is not None
    ]
    if not job_ids:
        return dict[str, SlurmJobUsage]()
    try:
        return query_job_usage(job_ids=job_ids)
    except SlurmError as error:
        _LOGGER.warning('%s', error)
        return dict[str, SlurmJobUsage]()

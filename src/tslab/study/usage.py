"""What each selected run used, and what to size the next jobs of each group to."""

from collections.abc import Mapping, Sequence
from datetime import timedelta
import logging
import math
from typing import cast

from tslab.helpers.text import log_table
from tslab.jobs.slurm import (
    format_slurm_duration,
    format_slurm_memory,
    parse_slurm_memory,
    query_job_usage,
    SlurmJobUsage,
)
from tslab.study.context import Context
from tslab.study.machine import SlurmSettings
from tslab.study.report import resources_of
from tslab.study.state import RunReport

__all__ = ['log_usage', 'usage_document']

_LOGGER = logging.getLogger(__name__)

# why: the margin over the largest observed usage, so the next job fits
_MARGIN = 1.2

# why: the error prefixes that mark a run that used its share of the job by design
_PAUSED_PREFIXES = ('TrainingPaused:', 'JobTimeLimitSignal:')


def usage_document(*, context: Context) -> dict[str, object]:
    """The usage of every selected run with a SLURM job, and the sizing of each group.

    Args:
        context: The context of the selected study runs.

    Returns:
        The versioned study-usage document.

    Raises:
        SlurmError: If the ``sacct`` usage query fails.
    """
    usage = _usage_of(context=context)
    selected = [
        (report, usage.get(report.job_id))
        for report in context.reports
        if report.job_id is not None
    ]
    runs = [
        _run_entry(context=context, report=report, usage=usage_item)
        for report, usage_item in selected
    ]
    return {
        'schema': 'tslab.study-usage/1',
        'study': context.reference or context.study.name,
        'machine': context.machine_path.resolve().as_posix(),
        'runs': runs,
        'groups': _groups(context=context, selected=selected),
    }


def _usage_of(*, context: Context) -> dict[str, SlurmJobUsage]:
    job_ids = [report.job_id for report in context.reports if report.job_id is not None]
    return {} if not job_ids else query_job_usage(job_ids=job_ids)


def _groups(
    *, context: Context, selected: list[tuple[RunReport, SlurmJobUsage | None]]
) -> list[dict[str, object]]:
    order: list[SlurmSettings | None] = []
    members: dict[SlurmSettings | None, list[tuple[RunReport, SlurmJobUsage | None]]] = {}
    for report, usage in selected:
        slurm = context.resolved[report.run.run_path].slurm
        if slurm not in members:
            members[slurm] = []
            order.append(slurm)
        members[slurm].append((report, usage))
    return [_group_entry(slurm=slurm, members=members[slurm]) for slurm in order]


def _group_entry(
    *, slurm: SlurmSettings | None, members: list[tuple[RunReport, SlurmJobUsage | None]]
) -> dict[str, object]:
    cpus = slurm.cpus_per_task if slurm is not None else 1
    elapsed = [
        usage.elapsed
        for report, usage in members
        if not _is_paused(report=report) and usage is not None and usage.elapsed is not None
    ]
    peak = [
        usage.max_rss for _, usage in members if usage is not None and usage.max_rss is not None
    ]
    return {
        'runs': len(members),
        'slurm': resources_of(slurm=slurm),
        'suggested_time': (
            format_slurm_duration(duration=max(elapsed) * _MARGIN) if elapsed else None
        ),
        'suggested_mem_per_cpu': (
            format_slurm_memory(size=math.ceil(max(peak) * _MARGIN / cpus)) if peak else None
        ),
    }


def _run_entry(
    *, context: Context, report: RunReport, usage: SlurmJobUsage | None
) -> dict[str, object]:
    slurm = context.resolved[report.run.run_path].slurm
    return {
        'run_path': report.run.run_path,
        'job_id': report.job_id,
        'state': usage.state if usage is not None else report.job_state,
        'elapsed': _duration_text(duration=usage.elapsed if usage is not None else None),
        'time_limit': _duration_text(duration=usage.time_limit if usage is not None else None),
        'max_rss': usage.max_rss if usage is not None else None,
        'requested_memory': _requested_memory(slurm=slurm),
        'total_cpu_seconds': (
            int(usage.total_cpu.total_seconds())
            if usage is not None and usage.total_cpu is not None
            else None
        ),
        'alloc_cpus': usage.alloc_cpus if usage is not None else None,
        'gpus': usage.gpus if usage is not None else 0,
        'paused': _is_paused(report=report),
    }


def _requested_memory(*, slurm: SlurmSettings | None) -> int | None:
    """The requested memory in bytes of one run, from its resolved ``[slurm]``."""
    if slurm is None or slurm.mem_per_cpu is None:
        return None
    per_cpu = parse_slurm_memory(text=slurm.mem_per_cpu)
    return None if per_cpu is None else per_cpu * slurm.cpus_per_task


def _is_paused(*, report: RunReport) -> bool:
    return (report.reason or '').startswith(_PAUSED_PREFIXES)


def _duration_text(*, duration: timedelta | None) -> str | None:
    return None if duration is None else format_slurm_duration(duration=duration)


def log_usage(*, document: Mapping[str, object]) -> None:
    """Log the usage table, the paused runs, and one suggestion line per group.

    Args:
        document: The study-usage document from ``usage_document``.
    """
    runs = _mappings(document.get('runs'))
    log_table(
        logger=_LOGGER,
        header=('run', 'job', 'state', 'elapsed/limit', 'peak/requested', 'cpu time', 'gpus'),
        rows=[
            (
                str(run['run_path']),
                str(run['job_id']),
                str(run['state']),
                _pair(run['elapsed'], run['time_limit']),
                _pair(_memory(value=run['max_rss']), _memory(value=run['requested_memory'])),
                _text(_seconds(value=run['total_cpu_seconds'])),
                str(run['gpus']),
            )
            for run in runs
        ],
    )
    for run in runs:
        if run.get('paused'):
            _LOGGER.info('  %s paused: needs more than one job', run['run_path'])
    for group in _mappings(document.get('groups')):
        _LOGGER.info(
            '%s: suggest time=%s mem_per_cpu=%s',
            group['slurm'],
            group['suggested_time'],
            group['suggested_mem_per_cpu'],
        )


def _mappings(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _pair(first: object, second: object) -> str:
    return f'{_text(first)}/{_text(second)}'


def _memory(*, value: object) -> str | None:
    if value is None:
        return None
    # why: usage_document writes these fields as ints in bytes
    return format_slurm_memory(size=cast('int', value))


def _seconds(*, value: object) -> str | None:
    if value is None:
        return None
    # why: usage_document writes this field as an int of seconds
    return format_slurm_duration(duration=timedelta(seconds=cast('int', value)))


def _text(value: object) -> str:
    return '-' if value is None else str(value)

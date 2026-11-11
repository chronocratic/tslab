"""What each selected run used, and what to size the next jobs of each group to."""

from collections.abc import Mapping, Sequence
from datetime import timedelta
import logging
import math
from typing import cast

from tslab.lab.core.status import PAUSED_ERROR_PREFIXES
from tslab.lab.core.text import format_bytes, format_duration, log_table
from tslab.lab.study.context import Context
from tslab.lab.study.launchers.common.protocols import Backend
from tslab.lab.study.launchers.common.types import JobUsage, Resources
from tslab.lab.study.launchers.tables import backend_for
from tslab.lab.study.reporting.report import resources_of
from tslab.lab.study.reporting.state import RunReport

__all__ = ['log_usage', 'usage_document']

_LOGGER = logging.getLogger(__name__)

# why: the margin over the largest observed usage, so the next job fits
_MARGIN = 1.2


def usage_document(*, context: Context) -> dict[str, object]:
    """The usage of every selected run with a SLURM job, and the sizing of each group.

    Args:
        context: The context of the selected study runs.

    Returns:
        The study-usage document.

    Raises:
        LaunchError: If the usage query fails on the scheduler.
    """
    backend = backend_for(machine=context.machine)
    usage, gpu_ids = backend.usage(context=context)
    selected = [
        (report, usage.get(report.job_id))
        for report in context.reports
        if report.job_id is not None
    ]
    runs = [
        _run_entry(
            context=context, report=report, usage=usage_item, gpu_id=gpu_ids.get(report.job_id)
        )
        for report, usage_item in selected
    ]
    return {
        'schema': 'tslab.study-usage',
        'launcher': context.machine.launcher,
        'study': context.reference or context.study.name,
        'machine': context.machine_path.resolve().as_posix(),
        'runs': runs,
        'groups': _groups(context=context, backend=backend, selected=selected),
    }


def _groups(
    *, context: Context, backend: Backend, selected: list[tuple[RunReport, JobUsage | None]]
) -> list[dict[str, object]]:
    order: list[Resources] = []
    members: dict[Resources, list[tuple[RunReport, JobUsage | None]]] = {}
    for report, usage in selected:
        resources = context.resolved[report.run.run_path].resources
        if resources not in members:
            members[resources] = []
            order.append(resources)
        members[resources].append((report, usage))
    return [
        _group_entry(backend=backend, resources=resources, members=members[resources])
        for resources in order
    ]


def _group_entry(
    *, backend: Backend, resources: Resources, members: list[tuple[RunReport, JobUsage | None]]
) -> dict[str, object]:
    elapsed = [
        usage.elapsed
        for report, usage in members
        if not _is_paused(report=report) and usage is not None and usage.elapsed is not None
    ]
    peak = [
        usage.max_rss for _, usage in members if usage is not None and usage.max_rss is not None
    ]
    suggested_time, suggested_mem_per_cpu = backend.suggest_sizing(
        elapsed=max(elapsed) * _MARGIN if elapsed else None,
        peak_per_cpu=math.ceil(max(peak) * _MARGIN / resources.cpus) if peak else None,
    )
    return {
        'runs': len(members),
        'slurm': resources_of(resources=resources),
        'suggested_time': suggested_time,
        'suggested_mem_per_cpu': suggested_mem_per_cpu,
    }


def _run_entry(
    *, context: Context, report: RunReport, usage: JobUsage | None, gpu_id: str | None
) -> dict[str, object]:
    resources = context.resolved[report.run.run_path].resources
    return {
        'run_path': report.run.run_path.as_posix(),
        'job_id': report.job_id,
        'state': usage.state if usage is not None else report.job_state,
        'elapsed': _duration_text(duration=usage.elapsed if usage is not None else None),
        'time_limit': _duration_text(duration=usage.time_limit if usage is not None else None),
        'max_rss': usage.max_rss if usage is not None else None,
        'requested_memory': resources.requested_memory,
        'total_cpu_seconds': (
            int(usage.total_cpu.total_seconds())
            if usage is not None and usage.total_cpu is not None
            else None
        ),
        'alloc_cpus': usage.alloc_cpus if usage is not None else None,
        'gpus': usage.gpus if usage is not None else 0,
        'gpu_id': gpu_id,
        'paused': _is_paused(report=report),
    }


def _is_paused(*, report: RunReport) -> bool:
    return (report.reason or '').startswith(PAUSED_ERROR_PREFIXES)


def _duration_text(*, duration: timedelta | None) -> str | None:
    return None if duration is None else format_duration(seconds=duration.total_seconds())


def log_usage(*, document: Mapping[str, object], has_time_limit: bool) -> None:
    """Log the usage table, the paused runs, and one suggestion line per group.

    Args:
        document: The study-usage document from ``usage_document``.
        has_time_limit: Whether the jobs of this backend have a time limit; a missing
            one shows as ``-``, else as ``no limit``.
    """
    runs = _mappings(value=document.get('runs'))
    log_table(
        logger=_LOGGER,
        header=('run', 'job', 'state', 'elapsed/limit', 'peak/requested', 'cpu time', 'gpus'),
        rows=[
            (
                str(run['run_path']),
                str(run['job_id']),
                str(run['state']),
                _pair(
                    first=run['elapsed'],
                    second=run['time_limit'],
                    missing='-' if has_time_limit else 'no limit',
                ),
                _pair(
                    first=_memory(value=run['max_rss']),
                    second=_memory(value=run['requested_memory']),
                ),
                _text(value=_seconds(value=run['total_cpu_seconds'])),
                str(run['gpus']),
            )
            for run in runs
        ],
    )
    for run in runs:
        if run.get('paused'):
            _LOGGER.info('  %s paused: needs more than one job', run['run_path'])
    for group in _mappings(value=document.get('groups')):
        if group['suggested_time'] is None and group['suggested_mem_per_cpu'] is None:
            continue
        _LOGGER.info(
            '%s: suggest time=%s mem_per_cpu=%s',
            group['slurm'],
            group['suggested_time'],
            group['suggested_mem_per_cpu'],
        )


def _mappings(*, value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _pair(*, first: object, second: object, missing: str = '-') -> str:
    return f'{_text(value=first)}/{missing if second is None else second}'


def _memory(*, value: object) -> str | None:
    if value is None:
        return None
    # why: usage_document writes these fields as ints in bytes
    return format_bytes(size=cast('int', value))


def _seconds(*, value: object) -> str | None:
    if value is None:
        return None
    # why: usage_document writes this field as an int of seconds
    return format_duration(seconds=cast('int', value))


def _text(*, value: object) -> str:
    return '-' if value is None else str(value)

"""What each selected run used, and what to size the next jobs of each group to."""

from collections.abc import Mapping, Sequence
from datetime import timedelta
import logging
import math
from pathlib import Path
from typing import cast

from tslab.core.files import read_json_object
from tslab.core.text import log_table
from tslab.jobs.slurm import (
    format_slurm_duration,
    format_slurm_memory,
    parse_slurm_memory,
    query_job_usage,
    SlurmJobUsage,
)
from tslab.study.context import Context
from tslab.study.launchers import Resources
from tslab.study.machine import SlurmSettings
from tslab.study.plan import log_file_of, study_folder
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
    local = context.machine.launcher == 'local'
    usage, gpu_ids = _local_usage_of(context=context) if local else (_usage_of(context=context), {})
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
        'schema': 'tslab.study-usage/1',
        'launcher': context.machine.launcher,
        'study': context.reference or context.study.name,
        'machine': context.machine_path.resolve().as_posix(),
        'runs': runs,
        'groups': _groups(context=context, selected=selected),
    }


def _local_usage_of(*, context: Context) -> tuple[dict[str, SlurmJobUsage], dict[str, str | None]]:
    """The usage of the latest task of every local run, read from its usage file.

    A missing or unreadable file gives a usage with no numbers.
    """
    folder = study_folder(machine=context.machine, study=context.study)
    usage: dict[str, SlurmJobUsage] = {}
    gpu_ids: dict[str, str | None] = {}
    for report in context.reports:
        if report.job_id is None or report.task is None:
            continue
        grid = context.study.grids[report.run.grid]
        log_file = log_file_of(folder=folder, short_name=grid.short_name, index=report.task.index)
        document = _read_usage_file(path=log_file.with_name(log_file.name + '.usage.json'))
        usage[report.job_id] = _local_job_usage(document=document, state=report.state.value)
        gpu_ids[report.job_id] = _gpu_id_of(document=document)
    return usage, gpu_ids


def _read_usage_file(*, path: Path) -> dict[str, object]:
    try:
        return read_json_object(path=path)
    except (OSError, ValueError):
        return {}


def _seconds_of(*, document: Mapping[str, object], key: str) -> timedelta | None:
    value = document.get(key)
    return timedelta(seconds=value) if isinstance(value, int | float) else None


def _gpu_id_of(*, document: Mapping[str, object]) -> str | None:
    value = document.get('gpu_id')
    return None if value is None else str(value)


def _local_job_usage(*, document: Mapping[str, object], state: str) -> SlurmJobUsage:
    rss = document.get('max_rss_bytes')
    return SlurmJobUsage(
        state=state,
        elapsed=_seconds_of(document=document, key='elapsed_seconds'),
        time_limit=_seconds_of(document=document, key='timeout'),
        total_cpu=_seconds_of(document=document, key='cpu_seconds'),
        max_rss=rss if isinstance(rss, int) else None,
        alloc_cpus=None,
        gpus=0 if _gpu_id_of(document=document) is None else 1,
    )


def _usage_of(*, context: Context) -> dict[str, SlurmJobUsage]:
    job_ids = [report.job_id for report in context.reports if report.job_id is not None]
    return {} if not job_ids else query_job_usage(job_ids=job_ids)


def _groups(
    *, context: Context, selected: list[tuple[RunReport, SlurmJobUsage | None]]
) -> list[dict[str, object]]:
    order: list[Resources] = []
    members: dict[Resources, list[tuple[RunReport, SlurmJobUsage | None]]] = {}
    for report, usage in selected:
        resources = context.resolved[report.run.run_path].resources
        if resources not in members:
            members[resources] = []
            order.append(resources)
        members[resources].append((report, usage))
    entries = [_group_entry(resources=resources, members=members[resources]) for resources in order]
    if context.machine.launcher == 'local':
        # why: a local run has no memory request to size; only the time limit is suggested
        for entry in entries:
            entry['suggested_mem_per_cpu'] = None
    return entries


def _group_entry(
    *, resources: Resources, members: list[tuple[RunReport, SlurmJobUsage | None]]
) -> dict[str, object]:
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
        'slurm': resources_of(resources=resources),
        'suggested_time': (
            format_slurm_duration(duration=max(elapsed) * _MARGIN) if elapsed else None
        ),
        'suggested_mem_per_cpu': (
            format_slurm_memory(size=math.ceil(max(peak) * _MARGIN / resources.cpus))
            if peak
            else None
        ),
    }


def _run_entry(
    *, context: Context, report: RunReport, usage: SlurmJobUsage | None, gpu_id: str | None
) -> dict[str, object]:
    resources = context.resolved[report.run.run_path].resources
    slurm = resources if isinstance(resources, SlurmSettings) else None
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
        'gpu_id': gpu_id,
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
    local = document.get('launcher') == 'local'
    log_table(
        logger=_LOGGER,
        header=('run', 'job', 'state', 'elapsed/limit', 'peak/requested', 'cpu time', 'gpus'),
        rows=[
            (
                str(run['run_path']),
                str(run['job_id']),
                str(run['state']),
                _pair(run['elapsed'], run['time_limit'], missing='no limit' if local else '-'),
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
        if local:
            _LOGGER.info('local: suggest timeout=%s', group['suggested_time'])
            continue
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


def _pair(first: object, second: object, *, missing: str = '-') -> str:
    return f'{_text(first)}/{missing if second is None else second}'


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

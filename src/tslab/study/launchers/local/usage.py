"""Reading usage files written by local tasks."""

from collections.abc import Mapping
from datetime import timedelta
from pathlib import Path

from tslab.core.files import read_json_object
from tslab.study.context import Context
from tslab.study.launchers import SlurmJobUsage
from tslab.study.plan import log_file_of, study_folder

__all__ = ['usage']


def usage(*, context: Context) -> tuple[dict[str, SlurmJobUsage], dict[str, str | None]]:
    """The latest local task usage, read from its usage file, with GPU ids."""
    folder = study_folder(machine=context.machine, study=context.study)
    usage_by_job: dict[str, SlurmJobUsage] = {}
    gpu_ids: dict[str, str | None] = {}
    for report in context.reports:
        if report.job_id is None or report.task is None:
            continue
        grid = context.study.grids[report.run.grid]
        log_file = log_file_of(folder=folder, short_name=grid.short_name, index=report.task.index)
        document = _read_usage_file(path=log_file.with_name(log_file.name + '.usage.json'))
        usage_by_job[report.job_id] = _local_job_usage(document=document, state=report.state.value)
        gpu_ids[report.job_id] = _gpu_id_of(document=document)
    return usage_by_job, gpu_ids


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

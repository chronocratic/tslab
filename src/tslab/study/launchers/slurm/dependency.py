"""SLURM dependency strings for chained job groups."""

from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath

from tslab.enums import AfterCondition, Satisfy
from tslab.study.plan_after import AfterTarget

__all__ = ['dependency_text']

_DEPENDENCIES: Mapping[AfterCondition, str] = {
    AfterCondition.OK: 'afterok',
    AfterCondition.ANY: 'afterany',
    AfterCondition.FAILED: 'afternotok',
    AfterCondition.STARTED: 'after',
}


def dependency_text(
    *, after: Sequence[AfterTarget], upstream: Mapping[PurePosixPath, str]
) -> str | None:
    """Format dependency targets as one SLURM ``--dependency`` value.

    Args:
        after: The conditions declared by the group.
        upstream: Same-launch run paths to submitted SLURM task ids.

    Returns:
        Comma-separated dependency entries, or ``None`` when no task id exists.

    Raises:
        ValueError: If an ``ALL`` entry lacks a same-launch upstream task id.
    """
    entries = [
        entry
        for target in after
        if (entry := _entry_text(target=target, upstream=upstream)) is not None
    ]
    return ','.join(entries) or None


def _entry_text(*, target: AfterTarget, upstream: Mapping[PurePosixPath, str]) -> str | None:
    """Format one condition and its ids, preserving the entry satisfy rule."""
    unresolved = [path for path in target.run_paths if path not in upstream]
    if target.satisfy is Satisfy.ALL and unresolved:
        paths = ', '.join(path.as_posix() for path in unresolved)
        message = f'cannot submit dependency: no job id for same-launch upstream run(s): {paths}'
        raise ValueError(message)
    ids = (*target.job_tasks, *(upstream[path] for path in target.run_paths if path in upstream))
    if not ids:
        return None
    if target.corr:
        # why: aftercorr names the upstream array; each task waits for the task with its index
        return f'aftercorr:{ids[0].partition("_")[0]}'
    dependency = _DEPENDENCIES[target.condition]
    delay = (
        f'+{target.delay_minutes}'
        if target.condition is AfterCondition.STARTED and target.delay_minutes > 0
        else ''
    )
    if target.satisfy is Satisfy.ALL:
        return f'{dependency}:{":".join(f"{job_id}{delay}" for job_id in ids)}'
    return '?'.join(f'{dependency}:{job_id}{delay}' for job_id in ids)

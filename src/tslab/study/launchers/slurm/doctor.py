"""Readiness checks for SLURM commands and cluster limits."""

from collections.abc import Mapping, Sequence
from datetime import timedelta
import logging
import shutil

from tslab.jobs.slurm import (
    format_slurm_duration,
    gres_gpus,
    parse_slurm_duration,
    query_cluster_config,
    query_partition,
    SlurmError,
    SlurmNodeGroup,
)
from tslab.study.doctor import DoctorCheck
from tslab.study.machine import SlurmSettings

__all__ = ['cluster_checks', 'slurm_check']

_LOGGER = logging.getLogger(__name__)

_SLURM_COMMANDS = ('sbatch', 'squeue', 'sacct')


def slurm_check() -> DoctorCheck:
    """Check that submission, queue and accounting commands are available."""
    missing = [name for name in _SLURM_COMMANDS if shutil.which(name) is None]
    if missing:
        return DoctorCheck(
            name='slurm', status='fail', message=f'missing SLURM commands: {", ".join(missing)}'
        )
    return DoctorCheck(name='slurm', status='ok', message='SLURM commands found')


def cluster_checks(*, slurm: SlurmSettings | None) -> tuple[DoctorCheck, ...]:
    """Check the SLURM settings of a profile against the cluster.

    The checks read the cluster with ``scontrol`` and ``sinfo``. A missing or
    failing command warns; it never fails the doctor.

    Args:
        slurm: The base ``[slurm]`` table of the profile; ``None`` when the
            profile has no SLURM table, which skips the checks with a warning.

    Returns:
        The ``slurm-array`` and ``slurm-partition`` checks, or one warning
        when the profile has no ``[slurm]`` table.
    """
    if slurm is None:
        return (
            DoctorCheck(
                name='slurm-array',
                status='warn',
                message='no [slurm] table in the profile; the cluster checks are skipped',
            ),
        )
    config = _cluster_config()
    return (
        _array_check(settings=slurm, config=config),
        _partition_check(settings=slurm, config=config),
    )


def _cluster_config() -> Mapping[str, str] | None:
    """The cluster config, or ``None`` when scontrol did not answer."""
    try:
        return query_cluster_config()
    except SlurmError as error:
        # why: a missing or failing scontrol is a warning for the doctor, never a failure
        _LOGGER.warning('scontrol show config failed: %s', error)
        return None


def _array_check(*, settings: SlurmSettings, config: Mapping[str, str] | None) -> DoctorCheck:
    """The slurm-array check: the profile array size against MaxArraySize."""
    if config is None:
        message = 'scontrol did not answer; the array limit is not checked'
        return DoctorCheck(name='slurm-array', status='warn', message=message)
    max_size = _int_or_none(text=config.get('MaxArraySize', ''))
    if max_size is None:
        message = 'scontrol reports no MaxArraySize; the array limit is not checked'
        return DoctorCheck(name='slurm-array', status='warn', message=message)
    # why: an array runs the indices 0 to MaxArraySize - 1
    allowed = max_size - 1
    if settings.max_array_size > allowed:
        message = (
            f'max_array_size {settings.max_array_size} is above the cluster limit: '
            f'MaxArraySize {max_size} allows {allowed}'
        )
        return DoctorCheck(name='slurm-array', status='fail', message=message)
    message = (
        f'max_array_size {settings.max_array_size} is within the cluster limit: '
        f'MaxArraySize {max_size} allows {allowed}'
    )
    return DoctorCheck(name='slurm-array', status='ok', message=message)


def _int_or_none(*, text: str) -> int | None:
    try:
        return int(text)
    except ValueError:
        return None


def _partition_check(*, settings: SlurmSettings, config: Mapping[str, str] | None) -> DoctorCheck:
    """The slurm-partition check: one job against the node groups of the partition.

    One node group must hold the time and the resources of the job.
    """
    try:
        groups = query_partition(partition=settings.partition)
    except SlurmError as error:
        # why: a missing or failing sinfo is a warning for the doctor, never a failure
        _LOGGER.warning('sinfo failed for partition %s: %s', settings.partition, error)
        message = (
            f'sinfo did not answer for partition {settings.partition}; '
            'the node sizes are not checked'
        )
        return DoctorCheck(name='slurm-partition', status='warn', message=message)
    if not groups:
        message = (
            f'sinfo lists no node of partition {settings.partition}; the node sizes are not checked'
        )
        return DoctorCheck(name='slurm-partition', status='warn', message=message)
    fragments, warning = _unmet_limits(settings=settings, groups=groups, config=config)
    if fragments:
        return DoctorCheck(name='slurm-partition', status='fail', message='; '.join(fragments))
    if warning is not None:
        return DoctorCheck(name='slurm-partition', status='warn', message=warning)
    message = f'a node of partition {settings.partition} holds one job'
    return DoctorCheck(name='slurm-partition', status='ok', message=message)


def _unmet_limits(
    *, settings: SlurmSettings, groups: Sequence[SlurmNodeGroup], config: Mapping[str, str] | None
) -> tuple[list[str], str | None]:
    """The fail lines of the partition check, and a warn line when memory is unproven.

    The fail lines are empty when one node group holds the time and the resources of
    one job. The memory verdict is scoped to the groups that already hold the time,
    CPUs and GPUs of one job: the job fails when every one of them is reported with
    a memory size below the requirement, and the warn line is set when some of them
    is not reported with a memory size.
    """
    fragments: list[str] = []
    job_time = parse_slurm_duration(text=settings.time)
    limit = _largest_time_limit(groups=groups)
    if limit is not None and job_time > limit:
        fragments.append(
            f'time {settings.time} is above the limit of partition {settings.partition} '
            f'({format_slurm_duration(duration=limit)})'
        )
    # why: one task per GPU, so a GPU job runs max(gpus, 1) tasks of the settings
    tasks = max(settings.gpus, 1)
    cpus = settings.cpus_per_task * tasks
    memory = settings.requested_memory
    if memory is not None:
        memory *= tasks
    # why: without a cluster config the tracking policy is unknown, so memory must not fail
    tracked = config is not None and _tracks_memory(config=config)
    checked_memory = memory if tracked else None
    if _group_fits_all(
        groups=groups,
        time=job_time,
        cpus=cpus,
        memory=checked_memory,
        gpus=settings.gpus,
        gpu_type=settings.gpu_type,
    ):
        return fragments, _policy_warning(settings=settings, memory=memory, config=config)
    fragments += _unmet_resource_limits(
        settings=settings, groups=groups, cpus=cpus, memory=checked_memory
    )
    if fragments:
        return fragments, None
    # why: a candidate holds the time, CPUs and GPUs of one job; only memory can fail it
    candidates = [
        group
        for group in groups
        if (group.time_limit is None or job_time <= group.time_limit)
        and group.cpus >= cpus
        and gres_gpus(gres=group.gres, gpu_type=settings.gpu_type) >= settings.gpus
    ]
    if not candidates or all(group.memory_bytes is not None for group in candidates):
        # why: every single resource can fit, yet no one group holds the whole job
        fragments.append(_split_limit(settings=settings, cpus=cpus, memory=checked_memory))
        return fragments, None
    return fragments, _memory_warning(settings=settings, needed=checked_memory, config=config)


def _largest_time_limit(*, groups: Sequence[SlurmNodeGroup]) -> timedelta | None:
    """The largest time limit of the groups, or ``None`` when any group has no limit."""
    largest: timedelta | None = None
    for group in groups:
        if group.time_limit is None:
            return None
        if largest is None or group.time_limit > largest:
            largest = group.time_limit
    return largest


def _tracks_memory(*, config: Mapping[str, str]) -> bool:
    """Whether the cluster tracks memory, from SelectTypeParameters.

    A missing key counts as tracked: a node would still refuse a job
    that asks for more memory than it has.
    """
    parameters = config.get('SelectTypeParameters')
    if parameters is None:
        return True
    return 'MEMORY' in parameters.upper()


def _group_fits_all(  # noqa: PLR0913 # why: one keyword-only argument per job requirement
    *,
    groups: Sequence[SlurmNodeGroup],
    time: timedelta,
    cpus: int,
    memory: int | None,
    gpus: int,
    gpu_type: str | None,
) -> bool:
    """Whether one node group holds the time, CPUs, memory and GPUs of one job.

    A ``time_limit`` of ``None`` means the group has no time limit.
    """
    return any(
        (group.time_limit is None or time <= group.time_limit)
        and group.cpus >= cpus
        and (memory is None or (group.memory_bytes is not None and group.memory_bytes >= memory))
        and gres_gpus(gres=group.gres, gpu_type=gpu_type) >= gpus
        for group in groups
    )


def _unmet_resource_limits(
    *, settings: SlurmSettings, groups: Sequence[SlurmNodeGroup], cpus: int, memory: int | None
) -> list[str]:
    """The fail lines of the single resources: CPUs, memory and GPUs."""
    return [
        fragment
        for fragment in (
            _cpu_limit(partition=settings.partition, needed=cpus, groups=groups),
            _memory_limit(partition=settings.partition, needed=memory, groups=groups),
            _gpu_limit(
                partition=settings.partition,
                needed=settings.gpus,
                gpu_type=settings.gpu_type,
                groups=groups,
            ),
        )
        if fragment is not None
    ]


def _policy_warning(
    *, settings: SlurmSettings, memory: int | None, config: Mapping[str, str] | None
) -> str | None:
    """The warn line when the tracking policy is unknown but memory is required.

    ``None`` when the policy is known, or no memory requirement is set.
    """
    if config is not None or memory is None:
        return None
    return _memory_warning(settings=settings, needed=memory, config=config)


def _memory_warning(
    *, settings: SlurmSettings, needed: int | None, config: Mapping[str, str] | None
) -> str | None:
    """The warn line when the memory requirement of one job cannot be checked.

    ``None`` when there is no memory requirement to check.
    """
    if needed is None:
        return None
    if config is None:
        return f'scontrol did not answer; the memory requirement of {needed} bytes is not checked'
    return (
        f'sinfo reports no usable memory for partition {settings.partition}; '
        f'the memory requirement of {needed} bytes is not checked'
    )


def _cpu_limit(*, partition: str, needed: int, groups: Sequence[SlurmNodeGroup]) -> str | None:
    if any(group.cpus >= needed for group in groups):
        return None
    largest = max(group.cpus for group in groups)
    return f'no node of partition {partition} has {needed} CPUs; the largest has {largest}'


def _memory_limit(
    *, partition: str, needed: int | None, groups: Sequence[SlurmNodeGroup]
) -> str | None:
    """The fail line when every node reports memory and none holds the job.

    A node without a reported memory keeps the line out: the requirement is
    then unproven, not failed.
    """
    sizes = [size for size in (group.memory_bytes for group in groups) if size is not None]
    if needed is None or len(sizes) != len(groups) or any(size >= needed for size in sizes):
        return None
    largest = max(sizes)
    return (
        f'no node of partition {partition} has {needed} bytes of memory; the largest has {largest}'
    )


def _gpu_limit(
    *, partition: str, needed: int, gpu_type: str | None, groups: Sequence[SlurmNodeGroup]
) -> str | None:
    if any(gres_gpus(gres=group.gres, gpu_type=gpu_type) >= needed for group in groups):
        return None
    largest = max(gres_gpus(gres=group.gres, gpu_type=gpu_type) for group in groups)
    if gpu_type is not None:
        return (
            f'no node of partition {partition} has {needed} GPUs of type {gpu_type}; '
            f'the largest has {largest}'
        )
    return f'no node of partition {partition} has {needed} GPUs; the largest has {largest}'


def _split_limit(*, settings: SlurmSettings, cpus: int, memory: int | None) -> str:
    needs = [f'{cpus} CPUs']
    if memory is not None:
        needs.append(f'{memory} bytes of memory')
    if settings.gpus:
        type_note = f' of type {settings.gpu_type}' if settings.gpu_type is not None else ''
        needs.append(f'{settings.gpus} GPUs{type_note}')
    return f'no node of partition {settings.partition} holds one job of {", ".join(needs)}'

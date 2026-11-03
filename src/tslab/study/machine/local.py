"""The ``[local]`` table of a machine profile."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path
from typing import cast

from tslab.enums import LauncherKind
from tslab.jobs.local import GpuDiscovery, MPS_GPU
from tslab.study.machine.tables import from_table, MachineProfileError

__all__ = [
    'FOLLOWUP_KEYS',
    'TABLE',
    'LocalSettings',
    'check_local',
    'local_gpus',
    'settings_from',
    'timeout_seconds',
]

TABLE = LauncherKind.LOCAL
FOLLOWUP_KEYS = frozenset(('auto_retry',))
_DURATION_HINT = 'use an sbatch --time format: "M", "M:S", "H:M:S", "D-H", "D-H:M" or "D-H:M:S"'


@dataclass(frozen=True, kw_only=True)
class LocalSettings:
    """The ``[local]`` table: how a local machine runs the tasks of a launch.

    Attributes:
        timeout: The per-task time limit, in sbatch duration format, when set.
        signal_seconds: Seconds before the limit to send ``SIGUSR2``; zero disables it.
        gpus: The GPU ids the machine may use (NVIDIA ids, or the Apple GPU
            ``"mps"`` standing alone), or ``GpuDiscovery.AUTO`` (``"auto"`` in the
            file) to find them at launch.
        skip_busy_gpus: Whether to leave out GPUs with running compute processes.
        busy_gpu_memory_fraction: The fraction of a GPU memory in use that
            also counts it as busy; with the default 0.1, a GPU needs 90% of
            its memory free.
        max_parallel: Tasks at once when the resolved GPU list is empty.
        tasks_per_gpu: Tasks that share one GPU id at once.
        keep_awake: Whether to keep the machine awake while a launch runs.
    """

    timeout: str | None = None
    signal_seconds: int = 120
    gpus: tuple[int | str, ...] | GpuDiscovery = ()
    skip_busy_gpus: bool = False
    busy_gpu_memory_fraction: float = 0.1
    max_parallel: int = 1
    tasks_per_gpu: int = 1
    keep_awake: bool = True


def timeout_seconds(*, local: LocalSettings) -> float | None:
    """The per-task time limit of a local launch, in seconds.

    Args:
        local: The ``[local]`` table of a local machine profile.

    Returns:
        The limit in seconds, or ``None`` when the table has no ``timeout``.
    """
    if local.timeout is None:
        return None
    return _parse_duration(text=local.timeout).total_seconds()


def settings_from(*, table: Mapping[str, object], path: Path) -> LocalSettings:
    """Read and validate this profile's local settings table."""
    local_table = table.get(TABLE)
    decoded_table = local_table
    gpus: tuple[int | str, ...] | GpuDiscovery = ()
    if isinstance(local_table, dict) and 'gpus' in local_table:
        # why: the config codec reads an enum by member name; the file spells the choice by value
        gpus = local_gpus(value=local_table['gpus'], location='local.gpus', path=path)
        decoded_table = {name: item for name, item in local_table.items() if name != 'gpus'}
    local = from_table(
        cls=LocalSettings, table=decoded_table, location=TABLE, path=path, required=False
    )
    local = replace(local or LocalSettings(), gpus=gpus)
    check_local(local=local, table=cast('Mapping[str, object]', local_table or {}), path=path)
    return local


def _parse_duration(*, text: str) -> timedelta:
    """Parse an sbatch duration without importing scheduler code."""
    days_text, separator, clock = text.partition('-')
    has_days = bool(separator)
    if not has_days:
        clock = text
    try:
        days = int(days_text) if has_days else 0
        parts = [int(part) for part in clock.split(':')]
    except ValueError as error:
        message = f'invalid duration {text!r}; {_DURATION_HINT}'
        raise ValueError(message) from error
    match has_days, parts:
        case True, [hours]:
            clock_delta = timedelta(hours=hours)
        case True, [hours, minutes]:
            clock_delta = timedelta(hours=hours, minutes=minutes)
        case True, [hours, minutes, seconds]:
            clock_delta = timedelta(hours=hours, minutes=minutes, seconds=seconds)
        case False, [minutes]:
            clock_delta = timedelta(minutes=minutes)
        case False, [minutes, seconds]:
            clock_delta = timedelta(minutes=minutes, seconds=seconds)
        case False, [hours, minutes, seconds]:
            clock_delta = timedelta(hours=hours, minutes=minutes, seconds=seconds)
        case _:
            message = f'invalid duration {text!r}; {_DURATION_HINT}'
            raise ValueError(message)
    if days < 0 or any(part < 0 for part in parts):
        message = f'invalid duration {text!r}; {_DURATION_HINT}, without negative parts'
        raise ValueError(message)
    return timedelta(days=days) + clock_delta


def check_local(*, local: LocalSettings, table: Mapping[str, object], path: Path) -> None:
    """The [local] rules that span several keys or the raw table."""
    if local.timeout is not None:
        try:
            _parse_duration(text=local.timeout)
        except ValueError as error:
            message = f'{path}: local.timeout: {error}'
            raise MachineProfileError(message) from error
    timeout = timeout_seconds(local=local)
    if timeout is not None and local.signal_seconds > 0 and local.signal_seconds >= timeout:
        message = (
            f'{path}: local.signal_seconds ({local.signal_seconds}) must be smaller than '
            f'local.timeout ({local.timeout}); lower signal_seconds or set it to 0'
        )
        raise MachineProfileError(message)
    if isinstance(table.get('gpus'), list) and 'max_parallel' in table:
        message = (
            f'{path}: local.gpus is a list, so local.max_parallel is unused; remove max_parallel'
        )
        raise MachineProfileError(message)
    if local.gpus == () and local.tasks_per_gpu > 1:
        message = f'{path}: local.tasks_per_gpu needs local.gpus; remove tasks_per_gpu or set gpus'
        raise MachineProfileError(message)
    if local.gpus in ((), (MPS_GPU,)) and local.skip_busy_gpus:
        message = (
            f'{path}: local.skip_busy_gpus reads nvidia-smi and needs NVIDIA gpus; '
            'remove skip_busy_gpus or set gpus'
        )
        raise MachineProfileError(message)
    if not 0 < local.busy_gpu_memory_fraction <= 1:
        message = f'{path}: local.busy_gpu_memory_fraction must be above 0 and at most 1'
        raise MachineProfileError(message)


def local_gpus(*, value: object, location: str, path: Path) -> tuple[int | str, ...] | GpuDiscovery:
    """The ``gpus`` field: ``"auto"``, a list of non-negative ids, or exactly ``["mps"]``."""
    if value == GpuDiscovery.AUTO:
        return GpuDiscovery.AUTO
    if value == [MPS_GPU]:
        return (MPS_GPU,)
    invalid = (
        f'{path}: {location} must be "auto", a list of non-negative integers, or exactly ["mps"]'
    )
    if not isinstance(value, list):
        raise MachineProfileError(invalid)
    if MPS_GPU in value:
        message = (
            f'{path}: {location} must not hold "mps" next to ids: '
            'MPS is the only GPU on a Mac, so it must stand alone'
        )
        raise MachineProfileError(message)
    if any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in value):
        raise MachineProfileError(invalid)
    return cast('tuple[int | str, ...]', tuple(value))  # checked above: ids are non-negative ints

"""The ``[local]`` table of a machine profile."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast, Literal

from tslab.jobs.slurm import parse_slurm_duration
from tslab.study.machine.tables import MachineProfileError

__all__ = [
    'LocalSettings',
    'check_local',
    'local_gpus',
    'reject_local_followups',
    'reject_slurm',
    'timeout_seconds',
]


@dataclass(frozen=True, kw_only=True)
class LocalSettings:
    """The ``[local]`` table: how a local machine runs the tasks of a launch.

    Attributes:
        timeout: The per-task time limit, in sbatch duration format, when set.
        signal_seconds: Seconds before the limit to send ``SIGUSR2``; zero disables it.
        gpus: The GPU ids the machine may use (NVIDIA ids, or the Apple GPU
            ``"mps"`` standing alone), or ``"auto"`` to find them at launch.
        skip_busy_gpus: Whether to leave out GPUs with running compute processes.
        max_parallel: Tasks at once when the resolved GPU list is empty.
        tasks_per_gpu: Tasks that share one GPU id at once.
        keep_awake: Whether to keep the machine awake while a launch runs.
    """

    timeout: str | None = None
    signal_seconds: int = 120
    gpus: tuple[int | Literal['mps'], ...] | Literal['auto'] = ()
    skip_busy_gpus: bool = False
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
    return parse_slurm_duration(text=local.timeout).total_seconds()


def check_local(*, local: LocalSettings, table: Mapping[str, object], path: Path) -> None:
    """The [local] rules that span several keys or the raw table."""
    if local.timeout is not None:
        try:
            parse_slurm_duration(text=local.timeout)
        except ValueError as error:
            message = f'{path}: local.timeout: {error}'
            raise MachineProfileError(message) from error
    if isinstance(table.get('gpus'), list) and 'max_parallel' in table:
        message = (
            f'{path}: local.gpus is a list, so local.max_parallel is unused; remove max_parallel'
        )
        raise MachineProfileError(message)
    if local.gpus == () and local.tasks_per_gpu > 1:
        message = f'{path}: local.tasks_per_gpu needs local.gpus; remove tasks_per_gpu or set gpus'
        raise MachineProfileError(message)
    if local.gpus in ((), ('mps',)) and local.skip_busy_gpus:
        message = (
            f'{path}: local.skip_busy_gpus reads nvidia-smi and needs NVIDIA gpus; '
            'remove skip_busy_gpus or set gpus'
        )
        raise MachineProfileError(message)


def local_gpus(*, value: object, location: str, path: Path) -> list[int | str] | Literal['auto']:
    """The ``gpus`` field: ``"auto"``, a list of non-negative ids, or exactly ``["mps"]``."""
    if value == 'auto':
        return 'auto'
    if value == ['mps']:
        return cast('list[int | str]', value)
    if isinstance(value, list):
        if 'mps' in value:
            message = (
                f'{path}: {location} must not hold "mps" next to ids: '
                'MPS is the only GPU on a Mac, so it must stand alone'
            )
            raise MachineProfileError(message)
        if any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in value):
            message = (
                f'{path}: {location} must be "auto", a list of non-negative integers, '
                'or exactly ["mps"]'
            )
            raise MachineProfileError(message)
        return cast('list[int | str]', value)
    message = (
        f'{path}: {location} must be "auto", a list of non-negative integers, or exactly ["mps"]'
    )
    raise MachineProfileError(message)


def reject_slurm(*, table: dict[str, object], path: Path) -> None:
    """A local launcher has no scheduler: SLURM tables would be silently half-applied."""
    if 'slurm' in table:
        message = f'{path}: slurm is not used by launcher "local"; remove the [slurm] table'
        raise MachineProfileError(message)
    entries = table.get('overrides')
    if not isinstance(entries, list):
        return
    for position, entry in enumerate(entries):
        if isinstance(entry, dict) and 'slurm' in entry:
            message = (
                f'{path}: overrides[{position}].slurm is not used by launcher "local"; remove it'
            )
            raise MachineProfileError(message)


def reject_local_followups(*, table: dict[str, object], path: Path) -> None:
    """wandb_sync and time queue SLURM jobs; a local launch has no scheduler to queue them on."""
    followup = table.get('followup')
    if not isinstance(followup, dict):
        return
    for key in ('wandb_sync', 'time'):
        if key in followup:
            message = f'{path}: followup.{key} is SLURM only; remove it'
            raise MachineProfileError(message)

"""The ``[slurm]`` table of a machine profile."""

from collections.abc import Mapping, Sequence
import dataclasses
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import cast, TYPE_CHECKING

from tslab.lab.enums import LauncherKind
from tslab.lab.jobs.slurm import parse_slurm_duration, parse_slurm_memory
from tslab.lab.study.definition import matching_in_order
from tslab.lab.study.machine.tables import (
    check_keys,
    check_range,
    decode,
    from_table,
    MachineProfileError,
)

if TYPE_CHECKING:
    from tslab.lab.study.machine.profile import MachineOverride

__all__ = [
    'FOLLOWUP_KEYS',
    'TABLE',
    'SlurmSettings',
    'sbatch_extra',
    'settings_from',
    'slurm_for',
    'slurm_overrides',
]

TABLE = LauncherKind.SLURM
FOLLOWUP_KEYS = frozenset(('retry', 'chained_retry', 'wandb_sync', 'time'))
_SBATCH_NAME_PATTERN = re.compile(r'[A-Za-z][A-Za-z0-9_-]*')
_UNSUPPORTED_SBATCH = frozenset({'clusters', 'cluster'})


@dataclass(frozen=True, kw_only=True)
class SlurmSettings:
    """The ``[slurm]`` table: the resources and setup of one sbatch job.

    Attributes:
        partition: The required SLURM partition.
        time: The job time limit.
        qos: The quality of service, when set.
        cpus_per_task: CPUs allocated to each task.
        mem_per_cpu: Memory allocated to each CPU, when set.
        gpus: GPUs allocated to each task.
        gpu_type: The GPU type for the GRES request, when set.
        max_concurrent: The maximum number of concurrent array jobs.
        max_array_size: The maximum number of tasks in an array.
        sbatch_extra: Additional ``#SBATCH`` options, keyed by long name.
        node_local_tmp: Whether to use node-local temporary folders.
        submit_pause_seconds: Delay between job submissions.
        setup: Shell lines run before each task.
        signal_seconds: Seconds before the time limit to send ``SIGUSR2``; zero disables it.
    """

    partition: str
    time: str = '2-00:00:00'
    qos: str | None = None
    cpus_per_task: int = 1
    mem_per_cpu: str | None = None
    gpus: int = 0
    gpu_type: str | None = None
    max_concurrent: int = 100
    max_array_size: int = 1000
    sbatch_extra: tuple[tuple[str, str | int | bool], ...] = ()
    node_local_tmp: bool = True
    submit_pause_seconds: float = 0.5
    setup: tuple[str, ...] = ()
    signal_seconds: int = 120

    @property
    def cpus(self) -> int:
        """The CPUs allocated to each task."""
        return self.cpus_per_task

    @property
    def cpus_per_run(self) -> int:
        """The CPUs of one run; a SLURM task is one run."""
        return self.cpus_per_task

    @property
    def max_tasks_per_job(self) -> int | None:
        """The maximum number of tasks in one array."""
        return self.max_array_size

    @property
    def requested_memory(self) -> int | None:
        """The total memory in bytes one task requests, or ``None`` when unbounded."""
        return _requested_memory(settings=self)

    def signal_problem(self) -> str | None:
        """The signal timing problem when the signal leaves no training time.

        Returns:
            The problem message, or ``None`` when the timing is valid.
        """
        if self.signal_seconds <= 0:
            return None
        time_seconds = parse_slurm_duration(text=self.time).total_seconds()
        if self.signal_seconds < time_seconds:
            return None
        return (
            f'slurm.signal_seconds ({self.signal_seconds}) must be smaller than '
            f'slurm.time ({self.time}); lower signal_seconds or set it to 0'
        )

    def as_record(self) -> dict[str, object]:
        """The resource fields of the group, for the submission record."""
        return cast('dict[str, object]', dataclasses.asdict(self))

    def describe(self) -> str:
        """The resource line of the settings, for the dry run printout."""
        # why: the fields mirror SlurmSettings, named in display order
        parts = [f'partition={self.partition}', f'time={self.time}']
        if self.qos is not None:
            parts.append(f'qos={self.qos}')
        parts.append(f'cpus={self.cpus_per_task}')
        if self.mem_per_cpu is not None:
            parts.append(f'mem/cpu={self.mem_per_cpu}')
        if self.gpus:
            parts.append(f'gpus={self.gpus}')
        return ', '.join(parts)


def _requested_memory(*, settings: SlurmSettings) -> int | None:
    """The requested memory in bytes of one run, from the settings."""
    if settings.mem_per_cpu is None:
        return None
    per_cpu = parse_slurm_memory(text=settings.mem_per_cpu)
    return None if per_cpu is None else per_cpu * settings.cpus_per_task


def slurm_for(
    *,
    base: SlurmSettings | None,
    overrides: Sequence['MachineOverride'],
    grid_name: str,
    values: Mapping[str, str],
) -> SlurmSettings | None:
    """The base settings with every matching profile override applied in order."""
    if base is None:
        return None  # why: no [slurm] table means no scheduler; overrides cannot add one
    merged: dict[str, object] = {
        item.name: getattr(base, item.name) for item in dataclasses.fields(base)
    }
    order = matching_in_order(
        matches=[override.match for override in overrides], grid_name=grid_name, values=values
    )
    for position in order:
        update = overrides[position].slurm
        if 'sbatch_extra' in update:
            # why: the base is a SlurmSettings field; profile overrides decode to the same type
            base_extra = cast('tuple[tuple[str, str | int | bool], ...]', merged['sbatch_extra'])
            override_extra = cast(
                'tuple[tuple[str, str | int | bool], ...]', update['sbatch_extra']
            )
            merged['sbatch_extra'] = tuple(
                sorted((dict(base_extra) | dict(override_extra)).items())
            )
        merged.update({key: value for key, value in update.items() if key != 'sbatch_extra'})
    return SlurmSettings(**merged)  # ty: ignore[invalid-argument-type]


def settings_from(*, table: Mapping[str, object], path: Path) -> SlurmSettings:
    """Read and validate this profile's SLURM settings table."""
    slurm_table = table.get(TABLE)
    if isinstance(slurm_table, dict) and 'sbatch_extra' in slurm_table:
        slurm_table = {
            **slurm_table,
            'sbatch_extra': [
                list(pair)
                for pair in sbatch_extra(
                    value=slurm_table['sbatch_extra'], location='slurm.sbatch_extra', path=path
                )
            ],
        }
    slurm = from_table(
        cls=SlurmSettings, table=slurm_table, location=TABLE, path=path, required=True
    )
    if slurm is None:
        message = f'{path}: the {TABLE} table is required'
        raise MachineProfileError(message)
    return slurm


def _converted_line(*, line: object) -> str:
    """One list entry of ``sbatch_extra`` as a line of the table form, or a comment."""
    fallback = f'# could not convert: {line}'
    if not (isinstance(line, str) and line.startswith('--')):
        return fallback
    name, separator, argument = line[2:].partition('=')
    if _SBATCH_NAME_PATTERN.fullmatch(name) is None:
        return fallback
    key = name.replace('-', '_')
    return f'{key} = {json.dumps(argument)}' if separator else f'{key} = true'


def sbatch_extra(
    *, value: object, location: str, path: Path
) -> tuple[tuple[str, str | int | bool], ...]:
    """The ``sbatch_extra`` table: sbatch long names to string, integer or boolean values."""
    if isinstance(value, list):
        converted = [_converted_line(line=line) for line in value]
        hint = '\n'.join(('[slurm.sbatch_extra]', *converted))
        message = f'{path}: {location} must be a table, not a list; use:\n{hint}'
        raise MachineProfileError(message)
    if not isinstance(value, dict):
        message = f'{path}: {location} must be a table'
        raise MachineProfileError(message)
    entries: dict[str, str | int | bool] = {}
    for key, item in value.items():
        if not _SBATCH_NAME_PATTERN.fullmatch(key):
            message = f'{path}: {location} key {key!r} must be an sbatch long name'
            raise MachineProfileError(message)
        name = key.replace('-', '_')
        if name in _UNSUPPORTED_SBATCH:
            # why: sbatch --parsable then prints jobid;cluster and every later call needs --clusters
            message = (
                f'{path}: {location} key {key!r}: multi-cluster submission is not supported; '
                'TSLab would not find the jobs (squeue, sacct and scancel ask the default cluster)'
            )
            raise MachineProfileError(message)
        if not isinstance(item, str | int | bool):
            message = f'{path}: {location}.{key} must be a string, integer, or boolean'
            raise MachineProfileError(message)
        check_range(key=key, value=item, location=f'{location}.{key}', path=path)
        if name in entries:
            message = f'{path}: {location} repeats {name!r} with both - and _'
            raise MachineProfileError(message)
        entries[name] = item
    return tuple(sorted(entries.items()))


def slurm_overrides(*, table: object, location: str, path: Path) -> dict[str, object]:
    """The keys of an override ``slurm`` table, each checked like the base ``[slurm]`` table."""
    if not isinstance(table, dict):
        message = f'{path}: {location}.slurm must be a table'
        raise MachineProfileError(message)
    check_keys(cls=SlurmSettings, table=table, location=f'{location}.slurm', path=path)
    if 'sbatch_extra' in table:
        table = {
            **table,
            'sbatch_extra': [
                list(pair)
                for pair in sbatch_extra(
                    value=table['sbatch_extra'],
                    location=f'{location}.slurm.sbatch_extra',
                    path=path,
                )
            ],
        }
    # why: decode against the full class; the partition of the probe is never read back
    probe = decode(
        data={'partition': 'probe', **table},
        cls=SlurmSettings,
        location=f'{location}.slurm',
        path=path,
    )
    values: dict[str, object] = {}
    for key in table:
        values[key] = getattr(probe, key)
        check_range(key=key, value=values[key], location=f'{location}.slurm.{key}', path=path)
    return values

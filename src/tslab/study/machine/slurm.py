"""The ``[slurm]`` table of a machine profile."""

from dataclasses import dataclass
import json
from pathlib import Path
import re

from tslab.study.machine.tables import check_keys, check_range, decode, MachineProfileError

__all__ = ['SlurmSettings', 'reject_local', 'sbatch_extra', 'slurm_overrides']

_SBATCH_NAME_PATTERN = re.compile(r'[A-Za-z][A-Za-z0-9_-]*')


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


def sbatch_extra(
    *, value: object, location: str, path: Path
) -> tuple[tuple[str, str | int | bool], ...]:
    """The ``sbatch_extra`` table: sbatch long names to string, integer or boolean values."""
    if isinstance(value, list):
        converted = []
        for line in value:
            if isinstance(line, str) and line.startswith('--'):
                name, separator, argument = line[2:].partition('=')
                if _SBATCH_NAME_PATTERN.fullmatch(name):
                    converted.append(
                        f'{name.replace("-", "_")} = {json.dumps(argument)}'
                        if separator
                        else f'{name.replace("-", "_")} = true'
                    )
                    continue
            converted.append(f'# could not convert: {line}')
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
        if not isinstance(item, str | int | bool):
            message = f'{path}: {location}.{key} must be a string, integer, or boolean'
            raise MachineProfileError(message)
        check_range(key=key, value=item, location=f'{location}.{key}', path=path)
        name = key.replace('-', '_')
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


def reject_local(*, table: dict[str, object], path: Path) -> None:
    """A SLURM launcher has no local settings: [local] tables would be silently half-applied."""
    if 'local' in table:
        message = f'{path}: local is not used by launcher "slurm"; remove the [local] table'
        raise MachineProfileError(message)
    entries = table.get('overrides')
    if not isinstance(entries, list):
        return
    for position, entry in enumerate(entries):
        if isinstance(entry, dict) and 'local' in entry:
            message = (
                f'{path}: overrides[{position}].local is not used by launcher "slurm"; remove it'
            )
            raise MachineProfileError(message)

"""SLURM job scripts for study jobs."""

from collections.abc import Mapping, Sequence
from pathlib import Path
import re
import shlex

from tslab.study.machine import SlurmSettings

__all__ = ['array_spec', 'job_script']

_EMPTY_INDICES = 'indices must not be empty'
_CONTROL_PATTERN = re.compile(r'[\x00-\x1f\x7f]')
_ENV_NAME_PATTERN = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')


def array_spec(*, indices: Sequence[int]) -> str:
    """A SLURM array spec: sorted, deduplicated, consecutive indices as ranges.

    Args:
        indices: The array indices of one job, in any order.

    Returns:
        For example ``'0-3,7,9-10'``.

    Raises:
        ValueError: If ``indices`` is empty.
    """
    ordered = sorted(set(indices))
    if not ordered:
        raise ValueError(_EMPTY_INDICES)
    parts: list[str] = []
    start = previous = ordered[0]
    for index in ordered[1:]:
        if index == previous + 1:
            previous = index
            continue
        parts.append(f'{start}-{previous}' if previous > start else str(start))
        start = previous = index
    parts.append(f'{start}-{previous}' if previous > start else str(start))
    return ','.join(parts)


def _q21_block(*, array: bool) -> str:
    job_id = '${SLURM_JOB_ID}'
    if array:
        job_id = '${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}'
    return (
        f'TMPDIR=/tmp/tslab-{job_id}\n'
        'mkdir -p "$TMPDIR"\n'
        'trap \'rm -rf "$TMPDIR"\' EXIT\n'
        'export TMPDIR\n'
        'export WANDB_CACHE_DIR=$TMPDIR/wandb-cache\n'
        'export WANDB_CONFIG_DIR=$TMPDIR/wandb-config'
    )


def _task_section(*, array: bool, tasks_file: Path) -> str:
    # why: on shared storage the file written by the launcher can reach the node late
    wait = f'for _ in $(seq 60); do [ -s {shlex.quote(str(tasks_file))} ] && break; sleep 1; done\n'
    if array:
        return wait + (
            'TASK_LINE=$(sed -n "s/^${SLURM_ARRAY_TASK_ID} //p" '
            + shlex.quote(str(tasks_file))
            + ')\n'
            'if [ -z "$TASK_LINE" ]; then\n'
            '    echo "no task ${SLURM_ARRAY_TASK_ID} in "'
            + shlex.quote(str(tasks_file))
            + ' >&2\n'
            '    exit 1\n'
            'fi\n'
            'ATTEMPT=${TASK_LINE%% *}\n'
            'TASK_LINE=${TASK_LINE#* }\n'
            'echo "=== tslab attempt ${ATTEMPT} | job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}'
            ' | $(hostname) | $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="\n'
            'srun --unbuffered --kill-on-bad-exit=1 bash -c "exec $TASK_LINE"'
        )
    return wait + (
        'TASK_LINE=$(sed -n 1p ' + shlex.quote(str(tasks_file)) + ')\n'
        # why: like the array section, the line starts with the index and the attempt
        'TASK_LINE=${TASK_LINE#* }\n'
        'TASK_LINE=${TASK_LINE#* }\n'
        'echo "=== tslab job ${SLURM_JOB_ID} | $(hostname) | $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="\n'
        'srun --unbuffered --kill-on-bad-exit=1 bash -c "exec $TASK_LINE"'
    )


def _check_text(*, value: object, what: str) -> None:
    """Reject a line break or other control character: it would start a new script line."""
    if _CONTROL_PATTERN.search(str(value)):
        message = f'{what} must not hold a line break or other control character: {value!r}'
        raise ValueError(message)


def _check_job_values(  # noqa: PLR0913 # why: keyword-only private helper, one per checked value
    *,
    job_name: str,
    slurm: SlurmSettings,
    log_pattern: Path,
    dependency: str | None,
    begin: str | None,
    env: Mapping[str, str],
) -> None:
    for what, value in (
        ('job_name', job_name),
        ('partition', slurm.partition),
        ('qos', slurm.qos or ''),
        ('time', slurm.time),
        ('mem_per_cpu', slurm.mem_per_cpu or ''),
        ('gpu_type', slurm.gpu_type or ''),
        ('log_pattern', log_pattern),
        ('dependency', dependency or ''),
        ('begin', begin or ''),
        *((f'sbatch_extra.{key}', value) for key, value in slurm.sbatch_extra),
        *(('setup', line) for line in slurm.setup),
    ):
        _check_text(value=value, what=what)
    for name in env:
        if _ENV_NAME_PATTERN.fullmatch(name) is None:
            message = f'{name!r} is not a valid environment variable name'
            raise ValueError(message)


def _as_sbatch_flag(*, key: str, value: object) -> str:
    # why: sbatch reads quotes in a directive as grouping, so a space in a value stays one value
    flag = f'#SBATCH --{key.replace("_", "-")}'
    return flag if value is True else f'{flag}={shlex.quote(str(value))}'


def job_script(  # noqa: PLR0913 # why: keyword-only public API; one option per script line
    *,
    job_name: str,
    slurm: SlurmSettings,
    indices: Sequence[int] | None,
    tasks_file: Path,
    log_pattern: Path,
    project_root: Path,
    env: Mapping[str, str],
    dependency: str | None = None,
    begin: str | None = None,
) -> str:
    """The full ``.sbatch`` script of one study job.

    Args:
        job_name: The ``--job-name``; kept short so logs stay readable.
        slurm: The resource settings of the job.
        indices: The array indices, or ``None`` for a single (follow-up) job.
        tasks_file: The ``.tasks`` file the script reads its command from.
        log_pattern: The ``--output`` pattern; SLURM fills in ``%a``.
        project_root: The directory the command runs in.
        env: Variables to export before the command.
        dependency: A ``--dependency`` value, when given.
        begin: A ``--begin`` value, when given.

    Returns:
        The script text, ``#SBATCH`` lines first.

    Raises:
        ValueError: If a value written to an ``#SBATCH`` line holds a line
            break or other control character, or a name in ``env`` is not a
            shell variable name.
    """
    _check_job_values(
        job_name=job_name,
        slurm=slurm,
        log_pattern=log_pattern,
        dependency=dependency,
        begin=begin,
        env=env,
    )
    array = indices is not None
    options: dict[str, object] = {'job_name': job_name, 'partition': slurm.partition}
    if slurm.qos is not None:
        options['qos'] = slurm.qos
    options.update(
        time=slurm.time,
        nodes=1,
        ntasks_per_node=max(slurm.gpus, 1),
        cpus_per_task=slurm.cpus_per_task,
    )
    if slurm.mem_per_cpu is not None:
        options['mem_per_cpu'] = slurm.mem_per_cpu
    if slurm.gpus > 0:
        gpu = f'gpu:{slurm.gpus}'
        if slurm.gpu_type is not None:
            gpu = f'gpu:{slurm.gpu_type}:{slurm.gpus}'
        options['gres'] = gpu
    if indices is not None:
        spec = f'{array_spec(indices=indices)}%{slurm.max_concurrent}'
        options['array'] = spec
    options.update(output=log_pattern, open_mode='append', no_requeue=True)
    if slurm.signal_seconds > 0:
        options['signal'] = f'USR2@{slurm.signal_seconds}'
    if dependency is not None:
        options['dependency'] = dependency
    if begin is not None:
        options['begin'] = begin
    options.update((key.replace('-', '_'), value) for key, value in slurm.sbatch_extra)
    lines = [
        '#!/bin/bash',
        *(
            _as_sbatch_flag(key=key, value=value)
            for key, value in options.items()
            if value is not False
        ),
    ]
    lines += [
        'set -euo pipefail',
        f'cd {shlex.quote(str(project_root))}',
        # why: a killed job would lose the output still held in the buffer
        'export PYTHONUNBUFFERED=1',
    ]
    lines += [f'export {name}={shlex.quote(value)}' for name, value in env.items()]
    lines += list(slurm.setup)
    if slurm.node_local_tmp:
        lines.append(_q21_block(array=array))
    lines.append(_task_section(array=array, tasks_file=tasks_file))
    return '\n'.join(lines) + '\n'

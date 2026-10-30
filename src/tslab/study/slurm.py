"""SLURM job scripts and scheduler calls for study jobs."""

from collections.abc import Callable, Mapping, Sequence
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time

from tslab.study.machine import SlurmSettings

__all__ = [
    'QOS_LIMIT_MARKERS',
    'SlurmError',
    'array_spec',
    'job_script',
    'query_job_states',
    'sbatch',
    'scancel',
    'slurm_available',
]

#: Output that marks a QoS submission limit; sbatch backs off and retries.
QOS_LIMIT_MARKERS = ('QOSMaxSubmitJobPerUserLimit', 'Job violates accounting/QOS policy')

_EMPTY_INDICES = 'indices must not be empty'
_CONTROL_PATTERN = re.compile(r'[\x00-\x1f\x7f]')
_ENV_NAME_PATTERN = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')
#: What squeue prints for job ids it no longer knows; the accounting database answers then.
_UNKNOWN_JOB = 'Invalid job id specified'


class SlurmError(RuntimeError):
    """A scheduler call failed in a way that is not a QoS limit."""


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


_JOB_PREFIXES = ('SLURM_', 'SLURMD_', 'SRUN_', 'SBATCH_', 'SALLOC_')
_JOB_NAMES = frozenset(
    {'MASTER_ADDR', 'MASTER_PORT', 'RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE'}
)


def _sbatch_env(*, environ: Mapping[str, str]) -> dict[str, str]:
    # why: a follow-up job submitted from inside a job would inherit the allocation of that job
    return {
        name: value
        for name, value in environ.items()
        if name == 'SLURM_CONF' or not (name.startswith(_JOB_PREFIXES) or name in _JOB_NAMES)
    }


def _run(
    command: list[str], *, env: Mapping[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    try:
        # why: fixed scheduler command on PATH; the exit code is checked by the caller
        return subprocess.run(  # noqa: S603
            command, check=False, capture_output=True, text=True, env=env
        )
    except FileNotFoundError as error:
        message = f'{command[0]} is not on PATH; is this a machine with SLURM?'
        raise SlurmError(message) from error


def sbatch(*, script: Path, sleep: Callable[[float], None] = time.sleep) -> str:
    """Submit one job script; QoS limits are waited out, other errors raise.

    Args:
        script: The ``.sbatch`` file to submit.
        sleep: The wait between retries; injected for the tests.

    Returns:
        The job id SLURM printed before the first ``;``.

    Raises:
        SlurmError: If ``sbatch`` fails without a QoS limit marker.
    """
    delay = 10.0
    while True:
        completed = _run(['sbatch', '--parsable', str(script)], env=_sbatch_env(environ=os.environ))
        if completed.returncode == 0:
            return completed.stdout.split(';', 1)[0].strip()
        output = completed.stdout + completed.stderr
        if not any(marker in output for marker in QOS_LIMIT_MARKERS):
            message = f'sbatch failed for {script}:\n{output}'
            raise SlurmError(message)
        sleep(delay)
        delay = min(delay * 2, 900.0)


def scancel(*, job_ids: Sequence[str]) -> None:
    """Cancel jobs; a non-zero exit raises.

    Args:
        job_ids: The job ids to cancel; nothing is called for an empty list.

    Raises:
        SlurmError: If ``scancel`` exits non-zero.
    """
    if not job_ids:
        return
    completed = _run(['scancel', *job_ids])
    if completed.returncode != 0:
        message = f'scancel failed for {list(job_ids)}:\n{completed.stdout}{completed.stderr}'
        raise SlurmError(message)


def slurm_available() -> bool:
    """Whether the scheduler commands this module calls exist on ``PATH``."""
    return all(shutil.which(name) is not None for name in ('sbatch', 'scancel', 'squeue', 'sacct'))


_COMPRESSED_ARRAY_KEY = re.compile(r'(?P<job>\d+)_\[(?P<ranges>[\d,-]+)(?:%\d+)?\]')


def _expand_array_key(*, key: str) -> list[str]:
    # why: sacct prints pending array tasks as one range, for example ``123_[4-7,9%10]``
    match = _COMPRESSED_ARRAY_KEY.fullmatch(key)
    if match is None:
        return [key]
    keys: list[str] = []
    for part in match['ranges'].split(','):
        first, _, last = part.partition('-')
        keys += [f'{match["job"]}_{index}' for index in range(int(first), int(last or first) + 1)]
    return keys


def _pipe_lines(*, stdout: str) -> dict[str, str]:
    states: dict[str, str] = {}
    for line in stdout.splitlines():
        key, _, state = (part.strip() for part in line.partition('|'))
        if not key or not state:
            continue
        for expanded in _expand_array_key(key=key):
            states[expanded] = state.split()[0]
    return states


def query_job_states(*, job_ids: Sequence[str]) -> dict[str, str]:
    """The SLURM state of every task id of the given jobs.

    Args:
        job_ids: Array or single job ids; empty means no calls and ``{}``.

    Returns:
        ``{task id: state}`` with task ids ``<job>_<index>`` or ``<job>``;
        ``squeue`` wins over ``sacct`` for the same id, and unparseable
        lines are ignored.

    Raises:
        SlurmError: If ``squeue`` or ``sacct`` exits non-zero (squeue not
            knowing a purged job id is not an error): an empty answer from a
            failed call must not read as every job missing.
    """
    if not job_ids:
        return {}
    ids = ','.join(job_ids)
    squeue = _run(['squeue', '--array', '--jobs', ids, '--noheader', '--format=%i|%T'])
    if squeue.returncode != 0 and _UNKNOWN_JOB not in squeue.stderr:
        message = f'squeue failed for jobs {ids}:\n{squeue.stdout}{squeue.stderr}'
        raise SlurmError(message)
    states = _pipe_lines(stdout=squeue.stdout)
    sacct = _run(
        [
            'sacct',
            '--jobs',
            ids,
            '--noheader',
            '--parsable2',
            '--allocations',
            '--format=JobID,State',
        ]
    )
    if sacct.returncode != 0:
        message = f'sacct failed for jobs {ids}:\n{sacct.stdout}{sacct.stderr}'
        raise SlurmError(message)
    for key, value in _pipe_lines(stdout=sacct.stdout).items():
        states.setdefault(key, value)
    return states

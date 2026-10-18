"""SLURM job scripts and scheduler calls for study jobs."""

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
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
    if array:
        return (
            'TASK_LINE=$(sed -n "s/^${SLURM_ARRAY_TASK_ID} //p" '
            + shlex.quote(str(tasks_file))
            + ')\n'
            'if [ -z "$TASK_LINE" ]; then\n'
            '    echo "no task ${SLURM_ARRAY_TASK_ID} in ' + str(tasks_file) + '" >&2\n'
            '    exit 1\n'
            'fi\n'
            'ATTEMPT=${TASK_LINE%% *}\n'
            'TASK_LINE=${TASK_LINE#* }\n'
            'echo "=== tslab attempt ${ATTEMPT} | job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}'
            ' | $(hostname) | $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="\n'
            'eval "$TASK_LINE"'
        )
    return (
        'TASK_LINE=$(sed -n 1p ' + shlex.quote(str(tasks_file)) + ')\n'
        # why: like the array section, the line starts with the index and the attempt
        'TASK_LINE=${TASK_LINE#* }\n'
        'TASK_LINE=${TASK_LINE#* }\n'
        'echo "=== tslab job ${SLURM_JOB_ID} | $(hostname) | $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="\n'
        'eval "$TASK_LINE"'
    )


def job_script(  # noqa: PLR0913 # why: keyword-only public API; each line of the script is one option
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
    """
    array = indices is not None
    lines = [
        '#!/bin/bash',
        f'#SBATCH --job-name={job_name}',
        f'#SBATCH --partition={slurm.partition}',
    ]
    if slurm.qos is not None:
        lines.append(f'#SBATCH --qos={slurm.qos}')
    lines += [f'#SBATCH --time={slurm.time}', f'#SBATCH --cpus-per-task={slurm.cpus_per_task}']
    if slurm.mem_per_cpu is not None:
        lines.append(f'#SBATCH --mem-per-cpu={slurm.mem_per_cpu}')
    if slurm.gpus > 0:
        lines.append(f'#SBATCH --gres=gpu:{slurm.gpus}')
    if indices is not None:
        lines.append(f'#SBATCH --array={array_spec(indices=indices)}%{slurm.max_concurrent}')
    lines += [
        f'#SBATCH --output={log_pattern}',
        '#SBATCH --open-mode=append',
        '#SBATCH --no-requeue',
    ]
    if dependency is not None:
        lines.append(f'#SBATCH --dependency={dependency}')
    if begin is not None:
        lines.append(f'#SBATCH --begin={begin}')
    lines += [f'#SBATCH {line}' for line in slurm.sbatch_extra]
    lines += ['set -euo pipefail', f'cd {shlex.quote(str(project_root))}']
    lines += [f'export {name}={shlex.quote(value)}' for name, value in env.items()]
    if slurm.node_local_tmp:
        lines.append(_q21_block(array=array))
    lines.append(_task_section(array=array, tasks_file=tasks_file))
    return '\n'.join(lines) + '\n'


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 # why: fixed scheduler command on PATH; exit checked by the caller
        command, check=False, capture_output=True, text=True
    )


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
        completed = _run(['sbatch', '--parsable', str(script)])
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
    return shutil.which('squeue') is not None and shutil.which('sacct') is not None


def _pipe_lines(*, stdout: str) -> dict[str, str]:
    states: dict[str, str] = {}
    for line in stdout.splitlines():
        key, _, state = (part.strip() for part in line.partition('|'))
        if not key or not state:
            continue
        states[key] = state.split()[0]
    return states


def query_job_states(*, job_ids: Sequence[str]) -> dict[str, str]:
    """The SLURM state of every task id of the given jobs.

    Args:
        job_ids: Array or single job ids; empty means no calls and ``{}``.

    Returns:
        ``{task id: state}`` with task ids ``<job>_<index>`` or ``<job>``;
        ``squeue`` wins over ``sacct`` for the same id, and unparseable
        lines are ignored.
    """
    if not job_ids:
        return {}
    ids = ','.join(job_ids)
    squeue = _run(['squeue', '--array', '--jobs', ids, '--noheader', '--format=%i|%T'])
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
    for key, value in _pipe_lines(stdout=sacct.stdout).items():
        states.setdefault(key, value)
    return states

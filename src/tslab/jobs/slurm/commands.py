"""SLURM scheduler commands and submission environment helpers."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from tslab.jobs.slurm.memory import parse_slurm_memory
from tslab.jobs.slurm.timing import parse_sacct_duration

__all__ = [
    'QOS_LIMIT_MARKERS',
    'SlurmError',
    'SlurmJobUsage',
    'query_expected_starts',
    'query_job_states',
    'query_job_usage',
    'sbatch',
    'scancel',
    'slurm_available',
    'submission_environment',
]

#: Output that marks a QoS submission limit; sbatch backs off and retries.
QOS_LIMIT_MARKERS = ('QOSMaxSubmitJobPerUserLimit', 'Job violates accounting/QOS policy')

#: What squeue prints for job ids it no longer knows; the accounting database answers then.
_UNKNOWN_JOB = 'Invalid job id specified'

_JOB_PREFIXES = ('SLURM_', 'SLURMD_', 'SRUN_', 'SBATCH_', 'SALLOC_')
_JOB_NAMES = frozenset(
    {'MASTER_ADDR', 'MASTER_PORT', 'RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE'}
)
_COMPRESSED_ARRAY_KEY = re.compile(r'(?P<job>\d+)_\[(?P<ranges>[\d,-]+)(?:%\d+)?\]')

# why: the number of fields the usage query asks for, so a short or long line is skipped
_USAGE_FIELDS = 8
_Allocation = tuple[str, timedelta | None, timedelta | None, timedelta | None, int | None, int]


class SlurmError(RuntimeError):
    """A scheduler call failed in a way that is not a QoS limit."""


def submission_environment(*, environ: Mapping[str, str]) -> dict[str, str]:
    """Return a copy of environment without inherited job-specific variables.

    Args:
        environ: Environment variables to filter, usually ``os.environ``.

    Returns:
        A copy without inherited allocation variables, except ``SLURM_CONF``.
    """
    # why: a follow-up job submitted from inside a job would inherit the allocation of that job
    return {
        name: value
        for name, value in environ.items()
        if name == 'SLURM_CONF' or not (name.startswith(_JOB_PREFIXES) or name in _JOB_NAMES)
    }


def _run(
    *, command: list[str], env: Mapping[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    try:
        # why: fixed scheduler command on PATH; the exit code is checked by the caller
        return subprocess.run(  # noqa: S603
            command, check=False, capture_output=True, text=True, env=env
        )
    except FileNotFoundError as error:
        message = f'{command[0]} is not on PATH; is this a machine with SLURM?'
        raise SlurmError(message) from error


def sbatch(
    *,
    script: Path,
    sleep: Callable[[float], None] = time.sleep,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Submit one job script; QoS limits are waited out, other errors raise.

    Args:
        script: The ``.sbatch`` file to submit.
        sleep: The wait between retries; injected for the tests.
        environ: The environment to clean and submit with; ``os.environ``
            when omitted.

    Returns:
        The job id SLURM printed before the first ``;``.

    Raises:
        SlurmError: If ``sbatch`` fails without a QoS limit marker.
    """
    delay = 10.0
    while True:
        completed = _run(
            command=['sbatch', '--parsable', str(script)],
            env=submission_environment(environ=environ if environ is not None else os.environ),
        )
        if completed.returncode == 0:
            return completed.stdout.split(';', 1)[0].strip()
        output = completed.stdout + completed.stderr
        if not any(marker in output for marker in QOS_LIMIT_MARKERS):
            message = f'sbatch failed for {script}:\n{output}'
            raise SlurmError(message)
        sleep(delay)
        delay = min(delay * 2, 900.0)


def scancel(*, job_ids: Sequence[str], signal: str | None = None) -> None:
    """Cancel jobs, or send their steps a signal; a non-zero exit raises.

    Args:
        job_ids: The job ids to cancel; nothing is called for an empty list.
        signal: Send this signal (such as ``USR2``) to the steps of the jobs
            instead of cancelling them, like ``--signal=USR2`` without ``B:``.

    Returns:
        None.

    Raises:
        SlurmError: If ``scancel`` exits non-zero.
    """
    if not job_ids:
        return
    command = ['scancel']
    if signal is not None:
        command.append(f'--signal={signal}')
    command += list(job_ids)
    completed = _run(command=command)
    if completed.returncode != 0:
        name = f'scancel --signal={signal}' if signal is not None else 'scancel'
        message = f'{name} failed for {list(job_ids)}:\n{completed.stdout}{completed.stderr}'
        raise SlurmError(message)


def slurm_available() -> bool:
    """Return whether the scheduler commands this module calls exist on ``PATH``.

    Returns:
        Whether ``sbatch``, ``scancel``, ``squeue`` and ``sacct`` exist on ``PATH``.
    """
    return all(shutil.which(name) is not None for name in ('sbatch', 'scancel', 'squeue', 'sacct'))


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


def query_expected_starts(*, job_ids: Sequence[str]) -> dict[str, str | None]:
    """Return the estimated start time of each job, as ``squeue --start`` prints it.

    Args:
        job_ids: Array or single job ids; empty means no calls and ``{}``.

    Returns:
        ``{task id: start text or None}`` with task ids ``<job>_<index>`` or
        ``<job>``; ``N/A`` and an empty value become ``None``, other values are
        returned as printed.

    Raises:
        SlurmError: If ``squeue`` exits non-zero (squeue not knowing a purged
            job id is not an error).
    """
    if not job_ids:
        return {}
    ids = ','.join(job_ids)
    squeue = _run(
        command=['squeue', '--array', '--start', '--jobs', ids, '--noheader', '--format=%i|%S']
    )
    if squeue.returncode != 0 and _UNKNOWN_JOB not in squeue.stderr:
        message = f'squeue --start failed for jobs {ids}:\n{squeue.stdout}{squeue.stderr}'
        raise SlurmError(message)
    return _pipe_start_lines(stdout=squeue.stdout)


def _pipe_start_lines(*, stdout: str) -> dict[str, str | None]:
    starts: dict[str, str | None] = {}
    for line in stdout.splitlines():
        key, _, value = (part.strip() for part in line.partition('|'))
        if not key:
            continue
        start = None if value in ('', 'N/A') else value
        for expanded in _expand_array_key(key=key):
            starts[expanded] = start
    return starts


def query_job_states(*, job_ids: Sequence[str]) -> dict[str, str]:
    """Return the SLURM state of every task id of the given jobs.

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
    squeue = _run(command=['squeue', '--array', '--jobs', ids, '--noheader', '--format=%i|%T'])
    if squeue.returncode != 0 and _UNKNOWN_JOB not in squeue.stderr:
        message = f'squeue failed for jobs {ids}:\n{squeue.stdout}{squeue.stderr}'
        raise SlurmError(message)
    states = _pipe_lines(stdout=squeue.stdout)
    sacct = _run(
        command=[
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


@dataclass(frozen=True, kw_only=True)
class SlurmJobUsage:
    """What ``sacct`` reports about the usage of one task of a job.

    Attributes:
        state: The first word of the task state, such as ``COMPLETED``.
        elapsed: The wall-clock time of the task, or ``None``.
        time_limit: The time limit of the task, or ``None``.
        total_cpu: The CPU seconds used, or ``None``.
        max_rss: The peak memory in bytes over the steps, or ``None``.
        alloc_cpus: The allocated CPUs, or ``None``.
        gpus: The GPUs from ``gres/gpu`` in the allocation, else ``0``.
    """

    state: str
    elapsed: timedelta | None
    time_limit: timedelta | None
    total_cpu: timedelta | None
    max_rss: int | None
    alloc_cpus: int | None
    gpus: int


def query_job_usage(*, job_ids: Sequence[str]) -> dict[str, SlurmJobUsage]:
    """Return what ``sacct`` reports about the usage of each task of the jobs.

    Args:
        job_ids: Array or single job ids; empty means no calls and ``{}``.

    Returns:
        ``{task id: usage}`` with task ids ``<job>_<index>`` or ``<job>``;
        step rows are folded into the task before the dot, and ``max_rss``
        is the maximum over the steps of the task.

    Raises:
        SlurmError: If ``sacct`` exits non-zero.
    """
    if not job_ids:
        return {}
    ids = ','.join(job_ids)
    sacct = _run(
        command=[
            'sacct',
            '--jobs',
            ids,
            '--noheader',
            '--parsable2',
            '--format=JobID,State,Elapsed,Timelimit,TotalCPU,MaxRSS,AllocCPUS,AllocTRES',
        ]
    )
    if sacct.returncode != 0:
        message = f'sacct failed for jobs {ids}:\n{sacct.stdout}{sacct.stderr}'
        raise SlurmError(message)
    return _parse_usage_rows(stdout=sacct.stdout)


def _note_peak(*, peak: dict[str, int], task_id: str, max_rss: str) -> None:
    """Raise the peak memory of a task to the ``MaxRSS`` of one step row."""
    rss = parse_slurm_memory(text=max_rss)
    if rss is not None:
        peak[task_id] = max(peak.get(task_id, 0), rss)


def _parse_usage_rows(*, stdout: str) -> dict[str, SlurmJobUsage]:
    allocations: dict[str, _Allocation] = {}
    peak: dict[str, int] = {}
    for line in stdout.splitlines():
        parts = [part.strip() for part in line.split('|')]
        if len(parts) < _USAGE_FIELDS or not parts[0]:
            continue
        job_id, state, elapsed, time_limit, total_cpu, max_rss, alloc_cpus, alloc_tres = parts[
            :_USAGE_FIELDS
        ]
        if '.' in job_id:
            # why: MaxRSS is only on the step rows, never on the allocation row
            _note_peak(peak=peak, task_id=job_id.split('.', 1)[0], max_rss=max_rss)
            continue
        allocations[job_id] = (
            state.split()[0] if state else '',
            parse_sacct_duration(text=elapsed),
            parse_sacct_duration(text=time_limit),
            parse_sacct_duration(text=total_cpu),
            int(alloc_cpus) if alloc_cpus else None,
            _gres_gpus(alloc_tres=alloc_tres),
        )
    return {
        task_id: SlurmJobUsage(
            state=state,
            elapsed=elapsed,
            time_limit=time_limit,
            total_cpu=total_cpu,
            max_rss=peak.get(task_id),
            alloc_cpus=cpus,
            gpus=gpus,
        )
        for task_id, (state, elapsed, time_limit, total_cpu, cpus, gpus) in allocations.items()
    }


def _gres_gpus(*, alloc_tres: str) -> int:
    for part in alloc_tres.split(','):
        if part.startswith('gres/gpu='):
            return _int_or_zero(text=part.split('=', 1)[1])
    return 0


def _int_or_zero(*, text: str) -> int:
    try:
        return int(text)
    except ValueError:
        return 0

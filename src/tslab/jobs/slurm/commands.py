"""SLURM scheduler commands and submission environment helpers."""

from collections.abc import Callable, Mapping, Sequence
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

__all__ = [
    'QOS_LIMIT_MARKERS',
    'SlurmError',
    'query_job_states',
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
        completed = _run(
            command=['sbatch', '--parsable', str(script)],
            env=submission_environment(environ=os.environ),
        )
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

    Returns:
        None.

    Raises:
        SlurmError: If ``scancel`` exits non-zero.
    """
    if not job_ids:
        return
    completed = _run(command=['scancel', *job_ids])
    if completed.returncode != 0:
        message = f'scancel failed for {list(job_ids)}:\n{completed.stdout}{completed.stderr}'
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

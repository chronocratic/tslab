"""A catalog of read-only SLURM commands, scoped to the jobs of one study.

Each entry is data: a tool, its arguments with placeholders, and which job ids it
takes. Callers render it to a shell file or run it directly.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
import subprocess

from tslab.lab.jobs.slurm.commands import (
    _pipe_lines,
    _run,
    _UNKNOWN_JOB,
    QUERY_TIMEOUT_SECONDS,
    SlurmError,
)
from tslab.lab.jobs.slurm.states import QUEUED_JOB_STATES, RUNNING_JOB_STATES

__all__ = [
    'SLURM_COMMANDS',
    'CommandOutput',
    'CommandValues',
    'IdScope',
    'SlurmCommand',
    'command_lines',
    'run_slurm_command',
    'select_ids',
    'slurm_command',
]


_FALLBACK_NOTE = 'sacct did not answer; asking every recorded job id\n'


class IdScope(StrEnum):
    """Which job ids a command takes."""

    ALL = 'all'
    LIVE = 'live'  # parent ids with a queued or running task
    RUNNING = 'running'  # ids of running tasks, such as ``123_4``


@dataclass(frozen=True, kw_only=True)
class SlurmCommand:
    """One catalog entry.

    Attributes:
        name: ``<tool>-<variant>``.
        description: One line for the docs and ``slurm.sh``.
        tool: The SLURM executable.
        arguments: The arguments after the tool; ``{ids}`` (comma-joined), ``{id}``
            (with ``per_job``), ``{width}`` (name column width) and ``{start}``
            (sacct start date) are filled in.
        scope: Which ids the command takes.
        per_job: Run once per id instead of once for all.
    """

    name: str
    description: str
    tool: str
    arguments: tuple[str, ...]
    scope: IdScope
    per_job: bool = False


@dataclass(frozen=True, kw_only=True)
class CommandValues:
    """What fills the placeholders: all recorded job ids, name width, sacct start date.

    Attributes:
        job_ids: Every recorded job id.
        width: The width of the name column.
        start: The start date for ``sacct``.
    """

    job_ids: tuple[str, ...]
    width: int
    start: str


@dataclass(frozen=True, kw_only=True)
class CommandOutput:
    """What a command printed and how it exited.

    Attributes:
        returncode: The exit code.
        stdout: The standard output.
        stderr: The standard error.
    """

    returncode: int
    stdout: str
    stderr: str


def _squeue(name: str, description: str, *arguments: str) -> SlurmCommand:
    return SlurmCommand(
        name=name, description=description, tool='squeue', arguments=arguments, scope=IdScope.LIVE
    )


def _sacct(name: str, description: str, *arguments: str) -> SlurmCommand:
    return SlurmCommand(
        name=name,
        description=description,
        tool='sacct',
        arguments=('--jobs', '{ids}', '--starttime', '{start}', *arguments),
        scope=IdScope.ALL,
    )


SLURM_COMMANDS: tuple[SlurmCommand, ...] = (
    _squeue(
        'squeue-full-name',
        'Queued and running jobs with the full job name.',
        '--array',
        '--jobs',
        '{ids}',
        '--format=%.18i %.9P %.{width}j %.8u %.2t %.10M %.6D %R',
    ),
    _squeue(
        'squeue-long',
        'Queued and running jobs in the long format with the full job name.',
        '--array',
        '--jobs',
        '{ids}',
        '--long',
        '--format=%.18i %.9P %.{width}j %.8u %.8T %.10M %.9l %.6D %R',
    ),
    _squeue(
        'squeue-start',
        'Expected start times of the queued jobs.',
        '--array',
        '--jobs',
        '{ids}',
        '--start',
        '--format=%.18i %.9P %.{width}j %.8u %.2t %.19S %.6D %20Y %R',
    ),
    _squeue(
        'squeue-running',
        'Running tasks only, with the full job name.',
        '--array',
        '--jobs',
        '{ids}',
        '--states=RUNNING',
        '--format=%.18i %.9P %.{width}j %.8u %.2t %.10M %.6D %R',
    ),
    _squeue(
        'squeue-stock', 'Queued and running jobs in the stock squeue format.', '--jobs', '{ids}'
    ),
    _sacct(
        'sacct-full-name',
        'Every job of the study from the accounting database, full job name.',
        '--format=JobID,JobName%{width},Partition,Account,AllocCPUS,State,ExitCode',
    ),
    _sacct(
        'sacct-failed',
        'Jobs that failed, timed out, ran out of memory or lost their node.',
        '--allocations',
        '--state=FAILED,TIMEOUT,OUT_OF_MEMORY,NODE_FAIL',
        '--format=JobID,JobName%{width},State,ExitCode,Elapsed,NodeList',
    ),
    _sacct(
        'sacct-usage',
        'Time, CPU and memory use of every job of the study.',
        '--format=JobID,JobName%{width},Elapsed,TotalCPU,ReqMem,MaxRSS,State',
    ),
    SlurmCommand(
        name='seff-all',
        description='Efficiency summary of every job of the study.',
        tool='seff',
        arguments=('{id}',),
        scope=IdScope.ALL,
        per_job=True,
    ),
    SlurmCommand(
        name='sstat-live',
        description='Live CPU and memory use of the running tasks.',
        tool='sstat',
        arguments=(
            '--allsteps',
            '--jobs',
            '{ids}',
            '--format=JobID,NTasks,AveCPU,MaxRSS,MaxVMSize',
        ),
        scope=IdScope.RUNNING,
    ),
    SlurmCommand(
        name='sprio-jobs',
        description='Scheduling priority of the queued jobs.',
        tool='sprio',
        arguments=('--jobs', '{ids}'),
        scope=IdScope.LIVE,
    ),
    SlurmCommand(
        name='scontrol-show',
        description='Full controller record of each job still held by the controller.',
        tool='scontrol',
        arguments=('show', 'job', '{id}'),
        scope=IdScope.LIVE,
        per_job=True,
    ),
)


def slurm_command(*, name: str) -> SlurmCommand:
    """The catalog entry called ``name``; ``KeyError`` lists the valid names."""
    for command in SLURM_COMMANDS:
        if command.name == name:
            return command
    message = f'unknown SLURM command {name!r}; use one of: ' + ', '.join(
        command.name for command in SLURM_COMMANDS
    )
    raise KeyError(message)


def command_lines(
    *, command: SlurmCommand, ids: Sequence[str], values: CommandValues
) -> list[list[str]]:
    """The argument lists to run: one per id when ``per_job``, none when ``ids`` is empty."""
    if not ids:
        return []
    fixed = {'{width}': str(values.width), '{start}': values.start}

    def fill(*, extra: dict[str, str]) -> list[str]:
        replacements = {**fixed, **extra}
        arguments = []
        for argument in command.arguments:
            for key, value in replacements.items():
                argument = argument.replace(key, value)  # noqa: PLW2901
            arguments.append(argument)
        return [command.tool, *arguments]

    if command.per_job:
        return [fill(extra={'{id}': job_id}) for job_id in ids]
    return [fill(extra={'{ids}': ','.join(ids)})]


def select_ids(*, scope: IdScope, job_ids: Sequence[str]) -> tuple[list[str], bool]:
    """The ids a scope takes, and whether the fallback to all ids was used.

    ``live`` and ``running`` ask ``sacct``; when it fails (no accounting), every
    recorded id is returned and the flag is true.
    """
    if scope is IdScope.ALL:
        return list(job_ids), False
    command = [
        'sacct',
        '--jobs',
        ','.join(job_ids),
        '--noheader',
        '--parsable2',
        '--allocations',
        '--starttime',
        '1970-01-01',
        '--format=JobID,State',
    ]
    try:
        # why: one try, no retries; a person is waiting at a terminal
        process = _run(command=command, timeout=QUERY_TIMEOUT_SECONDS)
    except SlurmError:
        return list(job_ids), True
    if process.returncode != 0:
        return list(job_ids), True
    states = _pipe_lines(stdout=process.stdout)
    if scope is IdScope.RUNNING:
        return [key for key, state in states.items() if state in RUNNING_JOB_STATES], False
    live = QUEUED_JOB_STATES | RUNNING_JOB_STATES
    return list(
        dict.fromkeys(key.split('_')[0] for key, state in states.items() if state in live)
    ), False


def _command_error(
    *, error: SlurmError, command: SlurmCommand, stdout: Sequence[str]
) -> CommandOutput:
    cause = error.__cause__
    if isinstance(cause, subprocess.TimeoutExpired):
        return CommandOutput(
            returncode=124,
            stdout=''.join(stdout),
            stderr=f'{command.tool} did not answer within {QUERY_TIMEOUT_SECONDS:g} seconds\n',
        )
    if isinstance(cause, FileNotFoundError):
        return CommandOutput(
            returncode=127,
            stdout=''.join(stdout),
            stderr=f'{command.tool} is not installed on this machine\n',
        )
    raise error


def run_slurm_command(*, command: SlurmCommand, values: CommandValues) -> CommandOutput:
    """Run a catalog command for the jobs of a study; outputs are joined in order."""
    ids, fell_back = select_ids(scope=command.scope, job_ids=values.job_ids)
    returncode = 0
    stdout: list[str] = []
    stderr: list[str] = [_FALLBACK_NOTE] if fell_back else []
    for line in command_lines(command=command, ids=ids, values=values):
        try:
            # why: fixed scheduler tool on PATH from the catalog
            process = _run(command=line, timeout=QUERY_TIMEOUT_SECONDS)
        except SlurmError as error:
            return _command_error(error=error, command=command, stdout=stdout)
        stdout.append(process.stdout)
        stderr.append(process.stderr)
        code = process.returncode
        # why: without accounting every recorded id is asked; the controller forgot purged ones
        if code != 0 and fell_back and _UNKNOWN_JOB in process.stderr:
            code = 0
        returncode = returncode or code
    return CommandOutput(returncode=returncode, stdout=''.join(stdout), stderr=''.join(stderr))

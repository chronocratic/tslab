"""SLURM job-script text helpers."""

from collections.abc import Mapping, Sequence
from pathlib import Path
import re
import shlex

__all__ = [
    'array_spec',
    'check_script_text',
    'export_lines',
    'node_local_tmp_lines',
    'sbatch_directives',
    'srun_command',
    'wait_for_file_line',
]

_EMPTY_INDICES = 'indices must not be empty'
_CONTROL_PATTERN = re.compile(r'[\x00-\x1f\x7f]')
_ENV_NAME_PATTERN = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')
_SBATCH_NAME_PATTERN = re.compile(r'[A-Za-z][A-Za-z0-9_-]*')


def array_spec(*, indices: Sequence[int]) -> str:
    """Format sorted, deduplicated array indices as ranges.

    Args:
        indices: The indices assigned to one array job.

    Returns:
        The compact SLURM array specification.

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


def sbatch_directives(*, options: Mapping[str, object]) -> list[str]:
    """Format ordered options as ``#SBATCH`` lines.

    Args:
        options: SLURM long option names and values, in output order.

    Returns:
        One directive for each option, excluding options set to ``False``.

    Raises:
        ValueError: If an option name is invalid or a value contains a control character.
    """
    lines = []
    for key, value in options.items():
        if _SBATCH_NAME_PATTERN.fullmatch(key) is None:
            message = f'{key!r} is not a valid SLURM option name'
            raise ValueError(message)
        check_script_text(value=value, what=f'sbatch option {key}')
        if value is False:
            continue
        flag = f'#SBATCH --{key.replace("_", "-")}'
        # why: sbatch reads quotes in a directive as grouping, so a space in a value stays one value
        lines.append(flag if value is True else f'{flag}={shlex.quote(str(value))}')
    return lines


def check_script_text(*, value: object, what: str) -> None:
    """Reject control characters that can add lines to generated text.

    Args:
        value: The value to check.
        what: The name used in any error message.

    Raises:
        ValueError: If ``value`` contains a control character.
    """
    if _CONTROL_PATTERN.search(str(value)):
        message = f'{what} must not hold a line break or other control character: {value!r}'
        raise ValueError(message)


def export_lines(*, env: Mapping[str, str]) -> list[str]:
    """Format environment variables as quoted shell exports.

    Args:
        env: Environment names and values, in output order.

    Returns:
        One shell export line per environment variable.

    Raises:
        ValueError: If a name is not a shell variable name.
    """
    lines = []
    for name, value in env.items():
        if _ENV_NAME_PATTERN.fullmatch(name) is None:
            message = f'{name!r} is not a valid environment variable name'
            raise ValueError(message)
        lines.append(f'export {name}={shlex.quote(value)}')
    return lines


def node_local_tmp_lines(*, prefix: str, array: bool) -> list[str]:
    """Build shell lines for a job-specific temporary directory.

    Args:
        prefix: The directory name prefix under ``/tmp``.
        array: Whether the path must include the array task ID.

    Returns:
        Assignment, creation, cleanup trap, and export lines.

    Raises:
        ValueError: If ``prefix`` contains a control character.
    """
    check_script_text(value=prefix, what='prefix')
    job_id = '${SLURM_JOB_ID}'
    if array:
        job_id = '${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}'
    return [
        f'TMPDIR=/tmp/{shlex.quote(prefix)}-{job_id}',
        'mkdir -p "$TMPDIR"',
        'trap \'rm -rf "$TMPDIR"\' EXIT',
        'export TMPDIR',
    ]


def wait_for_file_line(*, path: Path, seconds: int) -> str:
    """Format a shell loop that waits for a non-empty file.

    Args:
        path: The file to wait for.
        seconds: The maximum number of one-second checks.

    Returns:
        The shell loop as one line.

    Raises:
        TypeError: If ``path`` is not a ``Path``.
    """
    if not isinstance(path, Path):
        message = 'path must be a Path'
        raise TypeError(message)
    return f'for _ in $(seq {seconds}); do [ -s {shlex.quote(str(path))} ] && break; sleep 1; done'


def srun_command(*, variable: str) -> str:
    """Format an ``srun`` command that executes a shell variable.

    Args:
        variable: The shell variable holding the command line.

    Returns:
        The command with unbuffered output and failure propagation enabled.

    Raises:
        ValueError: If ``variable`` is not a shell variable name.
    """
    if _ENV_NAME_PATTERN.fullmatch(variable) is None:
        message = f'{variable!r} is not a valid shell variable name'
        raise ValueError(message)
    return f'srun --unbuffered --kill-on-bad-exit=1 bash -c "exec ${variable}"'

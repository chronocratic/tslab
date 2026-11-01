"""The shell script that runs one study run in the foreground of a local machine."""

from collections.abc import Mapping, Sequence
from pathlib import Path
import shlex

from tslab.jobs.local import export_lines

__all__ = ['foreground_script']


def foreground_script(*, project_root: Path, env: Mapping[str, str], command: Sequence[str]) -> str:
    """The shell script that runs one study run in the terminal.

    Args:
        project_root: The directory the command runs in.
        env: Variables to export before the command.
        command: The command to run; it replaces the script via ``exec``.

    Returns:
        The script text.

    Raises:
        ValueError: If a name in ``env`` is not a shell variable name.
    """
    lines = [
        '#!/bin/bash',
        'set -euo pipefail',
        f'cd {shlex.quote(str(project_root))}',
        # why: a killed run would lose the output still held in the buffer
        'export PYTHONUNBUFFERED=1',
        *export_lines(env=env),
        f'exec {shlex.join(command)}',
    ]
    return '\n'.join(lines) + '\n'

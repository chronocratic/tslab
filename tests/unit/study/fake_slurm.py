"""Fake SLURM executables for the study tests.

Each fake command records ``{"command", "args", "stdin"}`` to ``calls.jsonl``
and answers from ``answers.json``: per command, a list of
``{"exit", "stdout", "stderr"}`` consumed in order, the last one repeating.
With no answers, ``sbatch`` answers job ids ``1000``, ``1001``, and the rest
exit 0 without output.
"""

from collections.abc import Mapping
import json
import os
from pathlib import Path
import stat
import sys

import pytest

__all__ = ['FakeSlurm', 'install_fake_slurm']

_COMMANDS = ('sbatch', 'scancel', 'sacct', 'squeue')

# ponytail: the script counts its prior calls in calls.jsonl instead of a
# state file; the fakes are never called concurrently.
_SCRIPT = """\
#!%s
import json
import pathlib
import sys

NAME = '%s'
HERE = pathlib.Path(__file__).resolve().parent
CALLS = HERE / 'calls.jsonl'
with CALLS.open('a') as handle:
    handle.write(json.dumps({'command': NAME, 'args': sys.argv[1:], 'stdin': ''}) + '\\n')
prior = sum(
    1
    for line in CALLS.read_text().splitlines()
    if line.strip() and json.loads(line)['command'] == NAME
) - 1
answers = json.loads((HERE / 'answers.json').read_text()).get(NAME, [])
if answers:
    answer = answers[min(prior, len(answers) - 1)]
    sys.stdout.write(answer.get('stdout', ''))
    sys.stderr.write(answer.get('stderr', ''))
    sys.exit(int(answer.get('exit', 0)))
if NAME == 'sbatch':
    sys.stdout.write(f'{{1000 + prior}};tslab\\n')
sys.exit(0)
"""


class FakeSlurm:
    """The fakes installed under one directory, plus their recorded calls."""

    def __init__(self, *, base: Path) -> None:
        self._base = base

    def calls(self, command: str) -> list[list[str]]:
        """The argument lists the command was called with, in order."""
        path = self._base / 'calls.jsonl'
        if not path.is_file():
            return []
        return [
            json.loads(line)['args']
            for line in path.read_text().splitlines()
            if line.strip() and json.loads(line)['command'] == command
        ]

    def set_answers(self, command: str, answers: list[dict[str, object]]) -> None:
        """Replace the answers of one command; they restart from the first."""
        path = self._base / 'answers.json'
        existing = json.loads(path.read_text()) if path.is_file() else {}
        existing[command] = answers
        path.write_text(json.dumps(existing))


def install_fake_slurm(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answers: Mapping[str, object]
) -> FakeSlurm:
    """Install the fake scheduler commands on ``PATH`` and return their handle.

    Args:
        tmp_path: The test's scratch directory the fakes live in.
        monkeypatch: The fixture that puts the fakes on ``PATH`` and reverts it.
        answers: Per-command answer lists to start with.
    """
    base = tmp_path / 'slurm'
    base.mkdir()
    (base / 'answers.json').write_text(json.dumps(dict(answers)))
    for name in _COMMANDS:
        script = base / name
        script.write_text(_SCRIPT % (sys.executable, name))
        script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv('PATH', f'{base}{os.pathsep}{os.environ["PATH"]}')
    return FakeSlurm(base=base)

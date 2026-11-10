"""Shared TOML snippets and helpers of the machine profile tests."""

import os
from pathlib import Path

from tslab.study.machine.load import load_machine
from tslab.study.machine.profile import MachineProfile

__all__ = ['BASE_SLURM', 'LOCAL', 'SLURM_TABLE', 'load_profile', 'write']

BASE_SLURM = 'launcher = "slurm"\noutput_root = "runs"\n'
LOCAL = 'launcher = "local"\noutput_root = "runs"\n'
SLURM_TABLE = '[slurm]\npartition = "p"\ntime = "1:00:00"\n'


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / 'm.toml'
    path.write_text(text)
    return path


def load_profile(*, path: Path, lenient_env: bool = False) -> MachineProfile:
    """Load a profile against the test process environment."""
    return load_machine(path=path, lenient_env=lenient_env, environment=os.environ)

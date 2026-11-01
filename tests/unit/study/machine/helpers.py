"""Shared TOML snippets and helpers of the machine profile tests."""

from pathlib import Path

__all__ = ['BASE_SLURM', 'LOCAL', 'SLURM_TABLE', 'write']

BASE_SLURM = 'launcher = "slurm"\noutput_root = "runs"\n'
LOCAL = 'launcher = "local"\noutput_root = "runs"\n'
SLURM_TABLE = '[slurm]\npartition = "p"\ntime = "1:00:00"\n'


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / 'm.toml'
    path.write_text(text)
    return path

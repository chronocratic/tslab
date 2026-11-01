"""Readiness checks for SLURM commands."""

import shutil

from tslab.study.doctor import DoctorCheck

__all__ = ['slurm_check']

_SLURM_COMMANDS = ('sbatch', 'squeue', 'sacct')


def slurm_check() -> DoctorCheck:
    """Check that submission, queue and accounting commands are available."""
    missing = [name for name in _SLURM_COMMANDS if shutil.which(name) is None]
    if missing:
        return DoctorCheck(
            name='slurm', status='fail', message=f'missing SLURM commands: {", ".join(missing)}'
        )
    return DoctorCheck(name='slurm', status='ok', message='SLURM commands found')

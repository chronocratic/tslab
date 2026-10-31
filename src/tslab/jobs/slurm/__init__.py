"""SLURM helpers for job timing."""

from tslab.jobs.slurm.timing import parse_slurm_duration, remaining_job_seconds

__all__ = ['parse_slurm_duration', 'remaining_job_seconds']

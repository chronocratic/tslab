"""tslab.jobs: general helpers to run jobs on SLURM and local machines.

Planned to move to its own distribution.
"""

from tslab.jobs.slurm import parse_slurm_duration, remaining_job_seconds

__all__ = ['parse_slurm_duration', 'remaining_job_seconds']

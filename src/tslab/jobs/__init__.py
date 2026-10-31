"""tslab.jobs: general helpers to run jobs on SLURM and local machines.

Planned to move to its own distribution.
"""

from tslab.jobs.slurm import (
    parse_slurm_duration,
    QOS_LIMIT_MARKERS,
    query_job_states,
    remaining_job_seconds,
    sbatch,
    scancel,
    slurm_available,
    SlurmError,
    submission_environment,
)

__all__ = [
    'QOS_LIMIT_MARKERS',
    'SlurmError',
    'parse_slurm_duration',
    'query_job_states',
    'remaining_job_seconds',
    'sbatch',
    'scancel',
    'slurm_available',
    'submission_environment',
]

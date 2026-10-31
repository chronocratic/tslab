"""SLURM helpers for job timing."""

from tslab.jobs.slurm.commands import (
    QOS_LIMIT_MARKERS,
    query_expected_starts,
    query_job_states,
    sbatch,
    scancel,
    slurm_available,
    SlurmError,
    submission_environment,
)
from tslab.jobs.slurm.script import (
    array_spec,
    check_script_text,
    export_lines,
    node_local_tmp_lines,
    sbatch_directives,
    srun_command,
    wait_for_file_line,
)
from tslab.jobs.slurm.timing import parse_slurm_duration, remaining_job_seconds

__all__ = [
    'QOS_LIMIT_MARKERS',
    'SlurmError',
    'array_spec',
    'check_script_text',
    'export_lines',
    'node_local_tmp_lines',
    'parse_slurm_duration',
    'query_expected_starts',
    'query_job_states',
    'remaining_job_seconds',
    'sbatch',
    'sbatch_directives',
    'scancel',
    'slurm_available',
    'srun_command',
    'submission_environment',
    'wait_for_file_line',
]

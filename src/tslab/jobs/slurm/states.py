"""The SLURM job states, and whether one of them ends a job."""

__all__ = [
    'FAILING_JOB_STATES',
    'INTERRUPTING_JOB_STATES',
    'QUEUED_JOB_STATES',
    'RUNNING_JOB_STATES',
    'job_ended',
]

#: SLURM states in which the job has not started yet.
QUEUED_JOB_STATES = frozenset({'PENDING', 'REQUEUED'})

#: SLURM states in which the job is actively working.
RUNNING_JOB_STATES = frozenset(
    {'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED', 'STAGE_OUT', 'SIGNALING', 'RESIZING'}
)

#: Ended SLURM states that read as an interrupted job, not a failed one.
INTERRUPTING_JOB_STATES = frozenset({'TIMEOUT', 'NODE_FAIL', 'PREEMPTED'})

#: Ended SLURM states that read as a failed job.
FAILING_JOB_STATES = frozenset({'OUT_OF_MEMORY', 'FAILED', 'CANCELLED', 'BOOT_FAIL', 'DEADLINE'})


def job_ended(*, job_state: str | None) -> bool:
    """Whether a SLURM state says the job is over; ``None`` (unknown) is not."""
    return (
        job_state is not None
        and job_state not in QUEUED_JOB_STATES
        and job_state not in RUNNING_JOB_STATES
    )

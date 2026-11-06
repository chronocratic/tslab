"""The flags a job state can carry."""

from enum import StrEnum

__all__ = ['JobFlag']


class JobFlag(StrEnum):
    """What a backend says a job state means for the state logic."""

    #: The job stopped at its time limit.
    TIME_LIMIT = 'time_limit'
    #: The machine, not the run, ended the job; the attempt does not count toward the cap.
    NOT_COUNTED = 'not_counted'
    #: The backend cancelled the job before the run started.
    CANCELLED = 'cancelled'
    #: The job is queued while waiting for a dependency.
    DEPENDENCY_PENDING = 'dependency_pending'
    #: The job was cancelled because its dependency can never be met.
    DEPENDENCY_NEVER = 'dependency_never'
    #: SLURM holds the job because it reached MaxBatchRequeue.
    REQUEUE_HELD = 'requeue_held'

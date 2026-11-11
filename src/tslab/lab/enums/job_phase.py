"""The phases of a job state."""

from enum import StrEnum

__all__ = ['JobPhase']


class JobPhase(StrEnum):
    """The phase of a job state; the words of the scheduler stay in ``JobStates.states``.

    ``ENDED`` is over with the cause unknown; ``UNKNOWN`` was not queried.
    """

    QUEUED = 'queued'
    RUNNING = 'running'
    INTERRUPTED = 'interrupted'
    FAILED = 'failed'
    COMPLETED = 'completed'
    ENDED = 'ended'
    UNKNOWN = 'unknown'

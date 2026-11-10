"""The states shown for the runs of a study."""

from enum import StrEnum

__all__ = ['StudyState']


class StudyState(StrEnum):
    """The state shown for one run of a study."""

    QUEUED = 'queued'
    WAITING = 'waiting'
    RUNNING = 'running'
    DONE = 'done'
    INTERRUPTED = 'interrupted'
    FAILED = 'failed'
    DEPENDENCY_FAILED = 'dependency_failed'
    MISSING = 'missing'

"""The states shown for the runs of a study."""

from enum import StrEnum

__all__ = ['StudyState']


class StudyState(StrEnum):
    """The state shown for one run of a study."""

    QUEUED = 'queued'
    RUNNING = 'running'
    DONE = 'done'
    INTERRUPTED = 'interrupted'
    FAILED = 'failed'
    MISSING = 'missing'

"""The states of a task in a local launch."""

from enum import StrEnum

__all__ = ['LocalTaskState']


class LocalTaskState(StrEnum):
    """The state of one task of a local launch; the text is saved in the pid file."""

    PENDING = 'pending'
    RUNNING = 'running'
    ENDED = 'ended'

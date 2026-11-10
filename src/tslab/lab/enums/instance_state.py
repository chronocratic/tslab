"""The lifecycle states of a stage instance."""

from enum import StrEnum

__all__ = ['InstanceState']


class InstanceState(StrEnum):
    """Lifecycle of one stage instance."""

    RUNNING = 'running'
    DONE = 'done'
    FAILED = 'failed'
    SKIPPED = 'skipped'

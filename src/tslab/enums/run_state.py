"""The lifecycle states of a run."""

from enum import StrEnum

__all__ = ['RunState']


class RunState(StrEnum):
    """Lifecycle of a whole run.

    ``FINISHED`` means the ``with Run`` block ended without an exception. It does
    not mean every instance is ``done``: failed instances are allowed.
    """

    RUNNING = 'running'
    FINISHED = 'finished'
    FAILED = 'failed'
    CRASHED = 'crashed'

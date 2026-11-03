"""The run states shown to a person."""

from enum import StrEnum

__all__ = ['ShownRunState']


class ShownRunState(StrEnum):
    """The state of a run as ``tslab status`` shows it.

    Same words as ``RunState``, plus ``INTERRUPTED`` for a run that says
    ``running`` but has no live process.
    """

    RUNNING = 'running'
    FINISHED = 'finished'
    FAILED = 'failed'
    CRASHED = 'crashed'
    INTERRUPTED = 'interrupted'

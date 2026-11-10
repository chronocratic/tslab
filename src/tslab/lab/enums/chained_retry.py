"""Retry choices for chained grids."""

from enum import StrEnum

__all__ = ['ChainedRetry']


class ChainedRetry(StrEnum):
    """The retry policy for grids that are chained."""

    MANUAL = 'manual'
    RESUBMIT = 'resubmit'
    REQUEUE = 'requeue'

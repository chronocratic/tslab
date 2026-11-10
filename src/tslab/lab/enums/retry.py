"""Retry choices for grids without dependencies."""

from enum import StrEnum

__all__ = ['Retry']


class Retry(StrEnum):
    """The retry policy for grids that are not chained."""

    MANUAL = 'manual'
    RESUBMIT = 'resubmit'

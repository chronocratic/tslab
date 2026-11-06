"""Ways to combine matched upstream runs."""

from enum import StrEnum

__all__ = ['Satisfy']


class Satisfy(StrEnum):
    """Whether all or any matched upstream runs must meet a condition."""

    ALL = 'all'
    ANY = 'any'

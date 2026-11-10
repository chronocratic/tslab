"""Whether a chained grid's upstream conditions are met, impossible or pending."""

from enum import StrEnum

__all__ = ['AfterVerdict']


class AfterVerdict(StrEnum):
    """The current verdict for a downstream dependency."""

    MET = 'met'
    NEVER = 'never'
    OPEN = 'open'

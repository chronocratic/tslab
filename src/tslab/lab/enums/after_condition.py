"""Conditions a chained grid uses to wait for upstream runs."""

from enum import StrEnum

__all__ = ['AfterCondition']


class AfterCondition(StrEnum):
    """The upstream run condition a grid waits for."""

    OK = 'ok'
    ANY = 'any'
    FAILED = 'failed'
    STARTED = 'started'

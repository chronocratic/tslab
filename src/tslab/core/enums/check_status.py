"""The result of one readiness check."""

from enum import StrEnum

__all__ = ['CheckStatus']


class CheckStatus(StrEnum):
    """The result of one doctor check."""

    OK = 'ok'
    WARN = 'warn'
    FAIL = 'fail'

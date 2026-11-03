"""The log levels a run accepts."""

from enum import StrEnum

__all__ = ['LogLevel']


class LogLevel(StrEnum):
    """The level of the log output of TSLab; the text is the name in ``logging``."""

    DEBUG = 'DEBUG'
    INFO = 'INFO'
    WARNING = 'WARNING'
    ERROR = 'ERROR'

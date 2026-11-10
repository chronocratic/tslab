"""The error a ``tslab`` command raises when the user typed something unusable."""

__all__ = ['UsageError']


class UsageError(Exception):
    """A usage problem: the message is shown without a traceback and the exit code is 2."""

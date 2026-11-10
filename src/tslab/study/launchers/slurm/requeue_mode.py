"""Modes for SLURM job requeue directives."""

from enum import StrEnum

__all__ = ['RequeueMode']


class RequeueMode(StrEnum):
    """Whether and how a SLURM job may requeue."""

    DISABLED = 'disabled'
    ENABLED = 'enabled'
    EXPEDITE = 'expedite'

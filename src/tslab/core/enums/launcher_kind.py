"""The kinds of launcher a machine profile can name."""

from enum import StrEnum

__all__ = ['LauncherKind']


class LauncherKind(StrEnum):
    """The launcher of a machine; the text is the key in the profile and the job records."""

    SLURM = 'slurm'
    LOCAL = 'local'

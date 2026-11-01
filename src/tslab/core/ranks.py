"""The ranks of a distributed job: which process writes, how many tasks run.

A multi-rank job runs the user script once per rank. Only rank zero writes
results; the rank is read from the environment variables the launcher and
Lightning set. ``RankAgreement`` broadcasts a stage decision made on rank zero
to the other ranks.
"""

import os
from typing import Protocol, runtime_checkable

__all__ = ['RankAgreement', 'is_writer_process']

_RANK_VARIABLES = ('RANK', 'SLURM_PROCID', 'LOCAL_RANK')


@runtime_checkable
class RankAgreement(Protocol):
    """Broadcast a stage decision from rank zero to the other ranks."""

    def agree_across_ranks(self, *, key: str, value: bool) -> bool:
        """Return the decision of rank zero, waiting for it on other ranks."""


def is_writer_process() -> bool:
    """Return ``True`` unless this process is a non-zero rank of a distributed job.

    Every rank variable that is set must be ``0``. Checking all of them covers
    both ``srun`` tasks (``SLURM_PROCID``) and processes started by Lightning
    inside one SLURM task (``LOCAL_RANK``).
    """
    return all(os.environ.get(variable, '0') == '0' for variable in _RANK_VARIABLES)

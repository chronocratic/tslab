"""The job record of the local launcher."""

from dataclasses import dataclass

from tslab.lab.study.records import JobRecord

__all__ = ['LocalJobRecord']


@dataclass(frozen=True, kw_only=True)
class LocalJobRecord(JobRecord):
    """One job submitted on the local machine: the shared fields only."""

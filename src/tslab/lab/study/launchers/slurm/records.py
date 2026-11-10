"""The job record of the SLURM launcher: the shared fields and the job name."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, override, Self, TYPE_CHECKING

from tslab.lab.enums import LauncherKind
from tslab.lab.study.records import JobRecord, TaskRecord

if TYPE_CHECKING:
    from tslab.lab.study.plan import JobGroup

__all__ = ['SlurmJobRecord']


@dataclass(frozen=True, kw_only=True)
class SlurmJobRecord(JobRecord):
    """One job submitted to SLURM.

    Args:
        job_name: The ``--job-name`` of the job; ``None`` in lines written
            before this field.
    """

    job_name: str | None = None

    @override
    @classmethod
    def for_group(
        cls,
        *,
        group: 'JobGroup',
        submission_id: str,
        time: str,
        launcher: LauncherKind,
        job_id: str,
        tasks: tuple[TaskRecord, ...],
    ) -> Self:
        """The record of a submitted group, with its job name."""
        record = super().for_group(
            group=group,
            submission_id=submission_id,
            time=time,
            launcher=launcher,
            job_id=job_id,
            tasks=tasks,
        )
        return replace(record, job_name=group.job_name)

    @override
    def to_json(self) -> dict[str, object]:
        """The JSON form; the job name only when known."""
        document = super().to_json()
        if self.job_name is not None:
            document['job_name'] = self.job_name
        return document

    @override
    @classmethod
    def from_document(cls, *, document: Mapping[str, Any]) -> Self:
        """A record from its JSON form, without the version check."""
        job_name = document.get('job_name')
        return replace(
            super().from_document(document=document),
            job_name=None if job_name is None else str(job_name),
        )

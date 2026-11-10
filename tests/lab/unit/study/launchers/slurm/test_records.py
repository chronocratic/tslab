"""The SLURM job record: the job name kept beside the shared fields."""

import json

from tslab.enums import LauncherKind
from tslab.study.launchers.slurm.records import SlurmJobRecord
from tslab.study.records import JobKind, JobRecord


def _record(*, job_name: str | None) -> SlurmJobRecord:
    return SlurmJobRecord(
        kind=JobKind.RUN,
        submission_id='s1',
        time='2026-11-02T10:00:00+00:00',
        launcher=LauncherKind.SLURM,
        job_id='1000',
        job_file=1,
        tasks=(),
        job_name=job_name,
    )


def test_slurm_record_round_trips_with_its_job_name() -> None:
    record = _record(job_name='aug_a_0')
    document = json.loads(json.dumps(record.to_json()))
    assert document['job_name'] == 'aug_a_0'
    assert JobRecord.from_json(document=document) == record


def test_slurm_line_without_job_name_reads_as_a_slurm_record() -> None:
    # why: lines written before the field have no job name
    document = _record(job_name=None).to_json()
    assert 'job_name' not in document
    again = JobRecord.from_json(document=document)
    assert isinstance(again, SlurmJobRecord)
    assert again.job_name is None

"""The local job record: the shared fields only."""

import json

from tslab.enums import LauncherKind
from tslab.study.launchers.local.records import LocalJobRecord
from tslab.study.records import JobKind, JobRecord


def test_local_record_round_trips_as_a_local_record() -> None:
    record = LocalJobRecord(
        kind=JobKind.RUN,
        submission_id='s1',
        time='2026-11-02T10:00:00+00:00',
        launcher=LauncherKind.LOCAL,
        job_id='host-123',
        job_file=1,
        tasks=(),
    )
    document = json.loads(json.dumps(record.to_json()))
    again = JobRecord.from_json(document=document)
    assert isinstance(again, LocalJobRecord)
    assert again == record

"""``submissions.jsonl``: job records, appended and read back."""

import fcntl
import json
import logging
import os
from pathlib import Path, PurePosixPath
import threading
from typing import Any, cast

import pytest

from tslab.lab.core.status import InstanceRecord, RunStatusSnapshot, utc_now_iso, write_status
from tslab.lab.enums import InstanceState, LauncherKind, RunState
from tslab.lab.study.code_snapshot import CodeSnapshot
from tslab.lab.study.launchers.tables import JOB_RECORDS
from tslab.lab.study.records import (
    append_job,
    JobKind,
    JobRecord,
    latest_tasks,
    ProgressSnapshot,
    read_jobs,
    submission_lock,
    take_progress,
    TaskRecord,
)


def task(**changes: object) -> TaskRecord:
    values: dict[str, Any] = {
        'index': 0,
        'grid': 'g',
        'run_path': PurePosixPath('g/a/1'),
        'command': ('python', 'classify', 'no-preset'),
        'extra_args': (),
        'resources': {},
        'attempt': 1,
        'progress': ProgressSnapshot(done_instances=0, newest_checkpoint_mtime=None),
    }
    values.update(changes)
    return TaskRecord(**values)


def job(**changes: object) -> JobRecord:
    values: dict[str, object] = {
        'kind': JobKind.RUN,
        'submission_id': 's1',
        'time': utc_now_iso(),
        'launcher': LauncherKind.SLURM,
        'job_id': '100',
        'job_file': 1,
        'tasks': (),
    }
    values.update(changes)
    record_type = JOB_RECORDS[cast('LauncherKind', values['launcher'])]
    return record_type(**values)  # ty: ignore[invalid-argument-type]


def test_the_shared_job_record_is_not_built_directly() -> None:
    with pytest.raises(TypeError, match='launcher'):
        JobRecord(
            kind=JobKind.RUN,
            submission_id='s1',
            time=utc_now_iso(),
            launcher=LauncherKind.SLURM,
            job_id='100',
            job_file=1,
            tasks=(),
        )


def test_task_record_run_path_round_trips_as_posix_text() -> None:
    record = task(run_path=PurePosixPath('g/a/1'))
    document = json.loads(json.dumps(record.to_json()))
    assert document['run_path'] == 'g/a/1'
    again = TaskRecord.from_json(document=document)
    assert again.run_path == PurePosixPath('g/a/1')
    assert json.dumps(again.to_json()) == json.dumps(record.to_json())


def test_append_and_read_round_trip(tmp_path: Path) -> None:
    path = tmp_path / '.study' / 'submissions.jsonl'
    append_job(
        path=path,
        record=job(job_id='100', tasks=(task(run_path=PurePosixPath('g/a/1'), attempt=1),)),
    )
    append_job(
        path=path,
        record=job(job_id='101', tasks=(task(run_path=PurePosixPath('g/a/1'), attempt=2),)),
    )
    jobs = read_jobs(path=path)
    assert [record.job_id for record in jobs] == ['100', '101']
    assert json.loads(path.read_text().splitlines()[0])['format_version'] == 1
    assert latest_tasks(jobs=jobs)[PurePosixPath('g/a/1')][1].attempt == 2


def test_partial_last_line_is_skipped(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / 'submissions.jsonl'
    append_job(path=path, record=job(job_id='100'))
    with path.open('a') as handle:
        handle.write('{"format_version": 1, "kind": "ru')
    with caplog.at_level(logging.WARNING, logger='tslab.lab.study'):
        assert [record.job_id for record in read_jobs(path=path)] == ['100']
    assert 'line 2' in caplog.text


def test_append_after_torn_line_keeps_the_new_record_readable(tmp_path: Path) -> None:
    path = tmp_path / 'submissions.jsonl'
    append_job(path=path, record=job(job_id='100'))
    with path.open('a') as handle:
        handle.write('{"format_version": 1, "kind": "ru')
    append_job(path=path, record=job(job_id='101'))
    assert [record.job_id for record in read_jobs(path=path)] == ['100', '101']


def test_append_loops_over_short_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / 'submissions.jsonl'
    real_write = os.write

    def short_write(descriptor: int, data: bytes | memoryview) -> int:
        return real_write(descriptor, bytes(data)[:7])

    monkeypatch.setattr(os, 'write', short_write)
    append_job(path=path, record=job(job_id='100'))
    monkeypatch.undo()
    assert [record.job_id for record in read_jobs(path=path)] == ['100']


def test_followups_are_not_latest_tasks(tmp_path: Path) -> None:
    jobs = [
        job(job_id='1', tasks=(task(run_path=PurePosixPath('g/a/1')),)),
        job(job_id='2', kind='followup', tasks=()),
    ]
    assert latest_tasks(jobs=jobs)[PurePosixPath('g/a/1')][0].job_id == '1'


def test_unknown_format_version_raises(tmp_path: Path) -> None:
    path = tmp_path / 'submissions.jsonl'
    with path.open('w') as handle:
        handle.write(json.dumps({'format_version': 99}) + '\n')
    with pytest.raises(ValueError, match='format_version'):
        read_jobs(path=path)


def test_take_progress(tmp_path: Path) -> None:
    run_directory = tmp_path / 'g' / 'a' / '1'
    (run_directory / 'train').mkdir(parents=True)
    checkpoint = run_directory / 'train' / 'last.ckpt'
    checkpoint.write_text('ckpt')
    os.utime(checkpoint, (1_700_000_000.0, 1_700_000_000.0))
    snapshot = RunStatusSnapshot(
        run_state=RunState.FINISHED,
        updated_at=utc_now_iso(),
        instances={
            'train': InstanceRecord(state=InstanceState.DONE),
            'evaluate': InstanceRecord(state=InstanceState.DONE),
            'attack': InstanceRecord(state=InstanceState.FAILED, error='ValueError: x'),
        },
    )
    write_status(path=run_directory / 'status.json', snapshot=snapshot)
    progress = take_progress(run_directory=run_directory)
    assert progress.done_instances == 2
    assert progress.newest_checkpoint_mtime == pytest.approx(1_700_000_000.0)
    assert take_progress(run_directory=tmp_path / 'missing') == ProgressSnapshot(
        done_instances=0, newest_checkpoint_mtime=None
    )


def test_take_progress_finds_checkpoints_outside_the_stage_folders(tmp_path: Path) -> None:
    (tmp_path / 'checkpoints').mkdir()
    (tmp_path / 'checkpoints' / 'last.ckpt').write_text('ckpt')
    snapshot = RunStatusSnapshot(
        run_state=RunState.RUNNING,
        updated_at=utc_now_iso(),
        instances={'train[default]': InstanceRecord(state=InstanceState.RUNNING)},
    )
    write_status(path=tmp_path / 'status.json', snapshot=snapshot)
    assert take_progress(run_directory=tmp_path).newest_checkpoint_mtime is not None


def test_take_progress_skips_a_checkpoint_that_vanishes_before_its_stat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / 'a').mkdir()
    (tmp_path / 'a' / 'last.ckpt').write_text('ckpt')
    original_stat = Path.stat

    def vanishing_stat(self: Path, **kwargs: Any) -> os.stat_result:  # noqa: ANN401 # why: mirrors Path.stat
        if self.name == 'last.ckpt':
            raise FileNotFoundError(self.name)
        return original_stat(self, **kwargs)

    monkeypatch.setattr(Path, 'is_file', lambda self: self.name == 'last.ckpt')
    monkeypatch.setattr(Path, 'stat', vanishing_stat)
    assert take_progress(run_directory=tmp_path).newest_checkpoint_mtime is None


def test_take_progress_counts_an_unreadable_status_as_no_finished_instances(tmp_path: Path) -> None:
    (tmp_path / 'status.json').write_text('{')
    progress = take_progress(run_directory=tmp_path)
    assert progress.done_instances == 0
    assert progress.newest_checkpoint_mtime is None


def test_take_progress_finds_checkpoints_without_a_status_file(tmp_path: Path) -> None:
    (tmp_path / 'checkpoints').mkdir()
    (tmp_path / 'checkpoints' / 'last.ckpt').write_text('ckpt')
    progress = take_progress(run_directory=tmp_path)
    assert progress.done_instances == 0
    assert progress.newest_checkpoint_mtime is not None


def test_read_jobs_skips_a_line_missing_a_key(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / 's.jsonl'
    append_job(path=path, record=job())
    document = job().to_json()
    del document['job_id']
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(document) + '\n')
    with caplog.at_level('WARNING'):
        assert len(read_jobs(path=path)) == 1
    assert 'skipping line 2' in caplog.text


def test_submission_lock_excludes_a_second_holder(tmp_path: Path) -> None:
    with (
        submission_lock(study_folder=tmp_path),
        (tmp_path / 'submit.lock').open('a') as other,
        pytest.raises(BlockingIOError),
    ):
        fcntl.flock(other.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    with (tmp_path / 'submit.lock').open('a') as other:
        fcntl.flock(other.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_submission_lock_says_when_it_waits(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    other = (tmp_path / 'submit.lock').open('a')
    fcntl.flock(other.fileno(), fcntl.LOCK_EX)
    threading.Timer(0.3, other.close).start()
    with caplog.at_level('INFO'), submission_lock(study_folder=tmp_path):
        pass
    assert caplog.text.count('waiting for another submission of this study to finish') == 1


def test_submission_lock_times_out_when_the_lock_stays_busy(tmp_path: Path) -> None:
    with (tmp_path / 'submit.lock').open('a') as other:
        fcntl.flock(other.fileno(), fcntl.LOCK_EX)
        with (
            pytest.raises(TimeoutError, match='stayed busy'),
            submission_lock(study_folder=tmp_path, timeout_seconds=0.2),
        ):
            pass


def test_job_record_without_code_snapshot_serializes_as_before(tmp_path: Path) -> None:
    document = job().to_json()
    assert 'code_snapshot' not in document
    record = JobRecord.from_json(document=document)
    assert record.code_snapshot is None
    (tmp_path / 'old.jsonl').write_text(json.dumps(document) + '\n', encoding='utf-8')
    assert read_jobs(path=tmp_path / 'old.jsonl')[0].code_snapshot is None


def test_job_record_code_snapshot_round_trip(tmp_path: Path) -> None:
    snapshot = CodeSnapshot(
        folder=tmp_path / '.tslab-code-snapshots' / '0123456789abcdef',
        hash='0123456789abcdef',
        commit='a' * 40,
        dirty=True,
        untracked_copied=False,
    )
    record = job(code_snapshot=snapshot)
    document = record.to_json()
    assert document['code_snapshot'] == snapshot.to_json()
    assert JobRecord.from_json(document=document) == record


def test_task_record_without_counters_reads_as_zero() -> None:
    document = task().to_json()
    del document['uncounted_attempts'], document['uncounted_in_row']
    record = TaskRecord.from_json(document=document)
    assert (record.uncounted_attempts, record.uncounted_in_row) == (0, 0)
    assert TaskRecord.from_json(document=record.to_json()) == record


def test_task_record_counters_round_trip() -> None:
    record = task(uncounted_attempts=2, uncounted_in_row=1)
    assert TaskRecord.from_json(document=record.to_json()) == record


def test_from_json_rejects_an_unknown_job_kind() -> None:
    document = job().to_json()
    document['kind'] = 'batch'
    with pytest.raises(ValueError, match="'batch' is not a valid JobKind"):
        JobRecord.from_json(document=document)


def test_from_json_rejects_an_unknown_launcher() -> None:
    document = job().to_json()
    document['launcher'] = 'kubernetes'
    with pytest.raises(ValueError, match="'kubernetes' is not a valid LauncherKind"):
        JobRecord.from_json(document=document)

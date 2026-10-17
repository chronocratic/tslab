"""``submissions.jsonl``: job records, appended and read back."""

import json
import logging
import os
from pathlib import Path

import pytest

from tslab.core.status import (
    InstanceRecord,
    InstanceState,
    RunState,
    RunStatusSnapshot,
    utc_now_iso,
    write_status,
)
from tslab.study import (
    append_job,
    JobRecord,
    latest_tasks,
    ProgressSnapshot,
    read_jobs,
    take_progress,
    TaskRecord,
)


def task(**changes: object) -> TaskRecord:
    values: dict[str, object] = {
        'index': 0,
        'grid': 'g',
        'run_path': 'g/a/1',
        'command': ('python', 'classify', 'no-preset'),
        'extra_args': (),
        'resources': {},
        'attempt': 1,
        'progress': ProgressSnapshot(done_instances=0, newest_checkpoint_mtime=None),
    }
    values.update(changes)
    return TaskRecord(**values)  # type: ignore[arg-type]


def job(**changes: object) -> JobRecord:
    values: dict[str, object] = {
        'kind': 'run',
        'submission_id': 's1',
        'time': utc_now_iso(),
        'launcher': 'slurm',
        'job_id': '100',
        'job_file': 1,
        'tasks': (),
    }
    values.update(changes)
    return JobRecord(**values)  # type: ignore[arg-type]


def test_append_and_read_round_trip(tmp_path: Path) -> None:
    path = tmp_path / '.study' / 'submissions.jsonl'
    append_job(path=path, record=job(job_id='100', tasks=(task(run_path='g/a/1', attempt=1),)))
    append_job(path=path, record=job(job_id='101', tasks=(task(run_path='g/a/1', attempt=2),)))
    jobs = read_jobs(path=path)
    assert [record.job_id for record in jobs] == ['100', '101']
    assert json.loads(path.read_text().splitlines()[0])['format_version'] == 1
    assert latest_tasks(jobs=jobs)['g/a/1'][1].attempt == 2


def test_partial_last_line_is_skipped(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / 'submissions.jsonl'
    append_job(path=path, record=job(job_id='100'))
    with path.open('a') as handle:
        handle.write('{"format_version": 1, "kind": "ru')
    with caplog.at_level(logging.WARNING, logger='tslab.study'):
        assert [record.job_id for record in read_jobs(path=path)] == ['100']
    assert 'line 2' in caplog.text


def test_followups_are_not_latest_tasks(tmp_path: Path) -> None:
    jobs = [
        job(job_id='1', tasks=(task(run_path='g/a/1'),)),
        job(job_id='2', kind='followup', tasks=()),
    ]
    assert latest_tasks(jobs=jobs)['g/a/1'][0].job_id == '1'


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

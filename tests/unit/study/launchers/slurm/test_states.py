"""SLURM state queries and cache updates."""

from typing import Literal

import pytest

from tslab.core.status import utc_now_iso
from tslab.study.launchers.slurm import states
from tslab.study.records import JobRecord, ProgressSnapshot, TaskRecord


def _job_record(
    *,
    job_id: str,
    kind: Literal['run', 'followup'],
    launcher: Literal['slurm', 'local'],
    tasks: tuple[TaskRecord, ...] = (),
) -> JobRecord:
    return JobRecord(
        kind=kind,
        submission_id=job_id,
        time=utc_now_iso(),
        launcher=launcher,
        job_id=job_id,
        job_file=1,
        tasks=tasks,
    )


def _task(*, index: int, run_path: str) -> TaskRecord:
    return TaskRecord(
        index=index,
        grid='g',
        run_path=run_path,
        command=('python', 'run.py'),
        extra_args=(),
        resources={},
        attempt=1,
        progress=ProgressSnapshot(done_instances=0, newest_checkpoint_mtime=None),
    )


def test_slurm_task_ids_exclude_local_job_records() -> None:
    jobs = [
        _job_record(
            job_id='7', kind='run', launcher='slurm', tasks=(_task(index=0, run_path='g/a/1'),)
        ),
        _job_record(
            job_id='local-123-1',
            kind='run',
            launcher='local',
            tasks=(_task(index=1, run_path='g/a/2'),),
        ),
    ]

    assert states.slurm_task_ids(jobs=jobs) == ['7_0']


def test_followup_job_ids_exclude_local_job_records() -> None:
    jobs = [
        _job_record(job_id='20', kind='followup', launcher='slurm'),
        _job_record(job_id='local-123-1', kind='followup', launcher='local'),
    ]

    assert states.followup_job_ids(jobs=jobs) == ['20']


def test_query_states_returns_new_cache_without_mutating_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ended = {'1000_0': 'COMPLETED'}
    queried: list[list[str]] = []

    def query_job_states(*, job_ids: list[str]) -> dict[str, str]:
        queried.append(job_ids)
        return {'1000_1': 'RUNNING', '1000_2': 'COMPLETED'}

    monkeypatch.setattr(states, 'query_job_states', query_job_states)

    raw_states, new_cache = states._query_states(  # noqa: SLF001 # why: test cache immutability
        job_ids=['1000_0', '1000_1', '1000_2'], ended_states=ended
    )

    assert raw_states == {'1000_0': 'COMPLETED', '1000_1': 'RUNNING', '1000_2': 'COMPLETED'}
    assert new_cache == {'1000_0': 'COMPLETED', '1000_2': 'COMPLETED'}
    assert ended == {'1000_0': 'COMPLETED'}
    assert queried == [['1000_1', '1000_2']]

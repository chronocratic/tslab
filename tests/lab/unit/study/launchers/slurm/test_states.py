"""SLURM state queries and cache updates."""

from pathlib import Path, PurePosixPath

import pytest

from tests.lab.unit.study.helpers import context, submit_study
from tests.lab.unit.study.launchers.slurm.helpers import skip_query_waits
from tests.support.fake_slurm import FakeSlurm
from tslab.lab.core.status import utc_now_iso
from tslab.lab.enums import LauncherKind
from tslab.lab.study.launchers.common.types import QueuedJob
from tslab.lab.study.launchers.slurm import states
from tslab.lab.study.launchers.tables import JOB_RECORDS
from tslab.lab.study.records import JobKind, JobRecord, ProgressSnapshot, TaskRecord


def _job_record(
    *, job_id: str, kind: JobKind, launcher: LauncherKind, tasks: tuple[TaskRecord, ...] = ()
) -> JobRecord:
    return JOB_RECORDS[launcher](
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
        run_path=PurePosixPath(run_path),
        command=('python', 'run.py'),
        extra_args=(),
        resources={},
        attempt=1,
        progress=ProgressSnapshot(done_instances=0, newest_checkpoint_mtime=None),
    )


def test_slurm_task_ids_exclude_local_job_records() -> None:
    jobs = [
        _job_record(
            job_id='7',
            kind=JobKind.RUN,
            launcher=LauncherKind.SLURM,
            tasks=(_task(index=0, run_path='g/a/1'),),
        ),
        _job_record(
            job_id='local-123-1',
            kind=JobKind.RUN,
            launcher=LauncherKind.LOCAL,
            tasks=(_task(index=1, run_path='g/a/2'),),
        ),
    ]

    assert states.slurm_task_ids(jobs=jobs) == ['7_0']


def test_followup_job_ids_exclude_local_job_records() -> None:
    jobs = [
        _job_record(job_id='20', kind=JobKind.FOLLOWUP, launcher=LauncherKind.SLURM),
        _job_record(job_id='local-123-1', kind=JobKind.FOLLOWUP, launcher=LauncherKind.LOCAL),
    ]

    assert states.followup_job_ids(jobs=jobs) == ['20']


def test_followup_job_ids_lists_only_sorted_slurm_followups() -> None:
    jobs = [
        _job_record(job_id='20', kind=JobKind.FOLLOWUP, launcher=LauncherKind.SLURM),
        _job_record(job_id='10', kind=JobKind.RUN, launcher=LauncherKind.SLURM),
        _job_record(job_id='local-123-1', kind=JobKind.FOLLOWUP, launcher=LauncherKind.LOCAL),
        _job_record(job_id='3', kind=JobKind.FOLLOWUP, launcher=LauncherKind.SLURM),
    ]

    assert states.followup_job_ids(jobs=jobs) == ['20', '3']


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


def test_expected_starts_maps_queued_runs_to_entries(project: Path, fake: FakeSlurm) -> None:
    submit_study(project=project)
    fake.set_answers(
        'squeue',
        [
            {'exit': 0, 'stdout': '1000_0|PENDING\n'},
            {'exit': 0, 'stdout': '1000_0|2026-11-01T10:00:00|Priority\n'},
        ],
    )
    fake.set_answers(
        'sacct', [{'exit': 0, 'stdout': '1000_1|COMPLETED\n1000_2|COMPLETED\n1000_3|COMPLETED\n'}]
    )
    ctx = context(project=project, machine_file='cluster.toml')
    starts = states.expected_starts(context=ctx)
    assert starts == {
        '1000_0': QueuedJob(start='2026-11-01T10:00:00', reason='Priority', stuck=False)
    }


def test_expected_starts_flags_a_stuck_reason(project: Path, fake: FakeSlurm) -> None:
    submit_study(project=project)
    fake.set_answers(
        'squeue',
        [
            {'exit': 0, 'stdout': '1000_0|PENDING\n'},
            {'exit': 0, 'stdout': '1000_0|N/A|(PartitionTimeLimit)\n'},
        ],
    )
    fake.set_answers(
        'sacct', [{'exit': 0, 'stdout': '1000_1|COMPLETED\n1000_2|COMPLETED\n1000_3|COMPLETED\n'}]
    )
    ctx = context(project=project, machine_file='cluster.toml')
    starts = states.expected_starts(context=ctx)
    assert starts == {'1000_0': QueuedJob(start=None, reason='(PartitionTimeLimit)', stuck=True)}


def test_expected_starts_asks_nothing_without_queued_runs(project: Path, fake: FakeSlurm) -> None:
    submit_study(project=project)
    fake.set_answers('squeue', [{'exit': 0, 'stdout': '1000_0|RUNNING\n'}])
    fake.set_answers(
        'sacct', [{'exit': 0, 'stdout': '1000_1|COMPLETED\n1000_2|COMPLETED\n1000_3|COMPLETED\n'}]
    )
    ctx = context(project=project, machine_file='cluster.toml')
    starts = states.expected_starts(context=ctx)
    assert starts is None
    assert len(fake.calls('squeue')) == 1


def test_expected_starts_logs_and_gives_none_when_squeue_fails(
    project: Path,
    fake: FakeSlurm,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skip_query_waits(monkeypatch=monkeypatch)
    submit_study(project=project)
    fake.set_answers(
        'squeue',
        [
            {'exit': 0, 'stdout': '1000_0|PENDING\n'},
            {'exit': 1, 'stderr': 'slurm_load_jobs error: Socket timed out'},
        ],
    )
    fake.set_answers(
        'sacct', [{'exit': 0, 'stdout': '1000_1|COMPLETED\n1000_2|COMPLETED\n1000_3|COMPLETED\n'}]
    )
    ctx = context(project=project, machine_file='cluster.toml')
    with caplog.at_level('WARNING', logger='tslab.lab.study.launchers.slurm.states'):
        starts = states.expected_starts(context=ctx)
    assert starts is None
    assert 'squeue' in caplog.text

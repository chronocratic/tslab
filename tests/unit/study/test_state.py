"""Study states: the decision over folders and SLURM job states."""

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
    collect_states,
    FolderFacts,
    JobRecord,
    ProgressSnapshot,
    read_folder,
    slurm_task_ids,
    study_state,
    StudyRun,
    StudyState,
    TaskRecord,
)


def instance(state: str, error: str | None = None) -> InstanceRecord:
    return InstanceRecord(state=InstanceState(state), error=error)


def facts(
    *,
    run_state: str | None = None,
    instances: dict[str, str | tuple[str, str]] | None = None,
    lock_live: bool = False,
    read_error: str | None = None,
    missing: bool = False,
) -> FolderFacts:
    if missing:
        return FolderFacts(snapshot=None, read_error=None, lock_live=lock_live)
    records = {}
    for name, value in (instances or {}).items():
        records[name] = instance(value) if isinstance(value, str) else instance(*value)
    snapshot = None
    if run_state is not None:
        snapshot = RunStatusSnapshot(
            run_state=RunState(run_state), updated_at=utc_now_iso(), instances=records
        )
    return FolderFacts(snapshot=snapshot, read_error=read_error, lock_live=lock_live)


def run(run_path: str) -> StudyRun:
    return StudyRun(grid='g', values={'a': run_path.split('/')[1]}, run_path=run_path, settings={})


def task(index: int, run_path: str, attempt: int) -> TaskRecord:
    return TaskRecord(
        index=index,
        grid='g',
        run_path=run_path,
        command=('python', 'classify', 'no-preset'),
        extra_args=(),
        resources={},
        attempt=attempt,
        progress=ProgressSnapshot(done_instances=0, newest_checkpoint_mtime=None),
    )


def job(job_id: str, tasks: tuple[TaskRecord, ...]) -> JobRecord:
    return JobRecord(
        kind='run',
        submission_id='s1',
        time=utc_now_iso(),
        launcher='slurm',
        job_id=job_id,
        job_file=1,
        tasks=tasks,
    )


@pytest.mark.parametrize(
    ('job_state', 'folder', 'expected'),
    [
        ('PENDING', facts(), StudyState.QUEUED),
        ('RUNNING', facts(), StudyState.RUNNING),
        (None, facts(run_state='running', lock_live=True), StudyState.RUNNING),
        (None, facts(run_state='finished', instances={'train': 'done'}), StudyState.DONE),
        (None, facts(run_state='running'), StudyState.INTERRUPTED),
        (None, facts(run_state='crashed'), StudyState.INTERRUPTED),
        ('TIMEOUT', facts(), StudyState.INTERRUPTED),
        (
            None,
            facts(
                run_state='finished',
                instances={'train': ('failed', 'TrainingTimeLimitReached: 23:30')},
            ),
            StudyState.INTERRUPTED,
        ),
        (None, facts(run_state='failed'), StudyState.FAILED),
        (None, facts(run_state='finished', instances={'attack': 'skipped'}), StudyState.FAILED),
        ('OUT_OF_MEMORY', facts(missing=True), StudyState.FAILED),
        ('COMPLETED', facts(missing=True), StudyState.MISSING),
        (None, facts(missing=True), StudyState.MISSING),
        (None, facts(read_error='JSONDecodeError: x'), StudyState.FAILED),
    ],
)
def test_state_table(job_state: str | None, folder: FolderFacts, expected: StudyState) -> None:
    assert study_state(job_state=job_state, folder=folder)[0] is expected


def test_ended_job_overrides_a_live_looking_lock() -> None:
    folder = facts(run_state='running', lock_live=True)
    assert study_state(job_state='TIMEOUT', folder=folder)[0] is StudyState.INTERRUPTED


def test_reason_is_the_first_error_line() -> None:
    folder = facts(run_state='failed', instances={'train': ('failed', 'ValueError: bad\nmore')})
    _, reason = study_state(job_state=None, folder=folder)
    assert reason == 'ValueError: bad'


def test_collect_states_reads_folders_and_records(tmp_path: Path) -> None:
    runs = [run('g/a/1'), run('g/a/2')]
    (tmp_path / 'g' / 'a' / '1').mkdir(parents=True)
    write_status(
        path=tmp_path / 'g' / 'a' / '1' / 'status.json',
        snapshot=RunStatusSnapshot(
            run_state=RunState.FINISHED,
            updated_at=utc_now_iso(),
            instances={'train': InstanceRecord(state=InstanceState.DONE)},
        ),
    )
    jobs = [job('7', (task(index=0, run_path='g/a/1', attempt=1),))]
    reports = collect_states(
        study_directory=tmp_path, runs=runs, jobs=jobs, job_states={'7_0': 'COMPLETED'}
    )
    assert [(r.state, r.attempt, r.job_id) for r in reports] == [
        (StudyState.DONE, 1, '7_0'),
        (StudyState.MISSING, 0, None),
    ]


def test_unreadable_status_is_reported_not_raised(tmp_path: Path) -> None:
    folder = tmp_path / 'g' / 'a' / '1'
    folder.mkdir(parents=True)
    (folder / 'status.json').write_text('{')
    report = collect_states(
        study_directory=tmp_path, runs=[run('g/a/1')], jobs=[], job_states=None
    )[0]
    assert report.state is StudyState.FAILED
    assert report.reason is not None


def test_read_folder_never_raises(tmp_path: Path) -> None:
    (tmp_path / 'status.json').write_text('{')
    read = read_folder(run_directory=tmp_path)
    assert read.snapshot is None
    assert read.read_error is not None
    assert read.lock_live is False


def test_lock_live_uses_is_run_active(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr('tslab.study.state.is_run_active', lambda **_: True)
    read = read_folder(run_directory=tmp_path)
    assert read.lock_live is True


def test_slurm_task_ids() -> None:
    jobs = [
        job(
            '7',
            (
                task(index=0, run_path='g/a/1', attempt=1),
                task(index=3, run_path='g/a/2', attempt=1),
            ),
        ),
        job('9', (task(index=3, run_path='g/a/2', attempt=2),)),
    ]
    assert slurm_task_ids(jobs=jobs) == ['7_0', '9_3']


def test_run_report_job_id_is_the_local_id(tmp_path: Path) -> None:
    local_task = task(index=0, run_path='g/a/1', attempt=1)
    local_job = JobRecord(
        kind='run',
        submission_id='s2',
        time=utc_now_iso(),
        launcher='local',
        job_id='local-1-0',
        job_file=1,
        tasks=(local_task,),
    )
    report = collect_states(
        study_directory=tmp_path, runs=[run('g/a/1')], jobs=[local_job], job_states=None
    )[0]
    assert report.job_id == 'local-1-0'
    assert report.attempt == 1

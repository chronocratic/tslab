"""Study states: the decision over folders and SLURM job states."""

from collections.abc import Callable
from pathlib import Path

import pytest

from tests.unit.study.helpers import make_job_states
from tslab.core.status import InstanceRecord, RunStatusSnapshot, utc_now_iso, write_status
from tslab.enums import InstanceState, JobFlag, JobPhase, LauncherKind, RunState, StudyState
from tslab.jobs.slurm import (
    FAILING_JOB_STATES,
    INTERRUPTING_JOB_STATES,
    QUEUED_JOB_STATES,
    RUNNING_JOB_STATES,
)
from tslab.study import StudyRun
from tslab.study.launchers.slurm.states import slurm_task_ids
from tslab.study.records import JobKind, JobRecord, ProgressSnapshot, TaskRecord
from tslab.study.reporting.state import (
    _grace_wait,
    collect_states,
    FolderFacts,
    read_folder,
    RunReport,
    study_state,
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
        kind=JobKind.RUN,
        submission_id='s1',
        time=utc_now_iso(),
        launcher=LauncherKind.SLURM,
        job_id=job_id,
        job_file=1,
        tasks=tasks,
    )


@pytest.mark.parametrize(
    ('job_reason', 'job_phase', 'folder', 'expected', 'reason'),
    [
        *(
            (None, JobPhase.QUEUED, facts(), StudyState.QUEUED, None)
            for state in sorted(QUEUED_JOB_STATES)
        ),
        *(
            (None, JobPhase.RUNNING, facts(), StudyState.RUNNING, None)
            for state in sorted(RUNNING_JOB_STATES)
        ),
        *(
            (
                f'SLURM {state}',
                JobPhase.INTERRUPTED,
                facts(),
                StudyState.INTERRUPTED,
                f'SLURM {state}',
            )
            for state in sorted(INTERRUPTING_JOB_STATES)
        ),
        *(
            (f'SLURM {state}', JobPhase.FAILED, facts(), StudyState.FAILED, f'SLURM {state}')
            for state in sorted(FAILING_JOB_STATES)
        ),
        (
            None,
            JobPhase.UNKNOWN,
            facts(run_state='running', lock_live=True),
            StudyState.RUNNING,
            None,
        ),
        (
            None,
            JobPhase.UNKNOWN,
            facts(run_state='finished', instances={'train': 'done'}),
            StudyState.DONE,
            None,
        ),
        (None, JobPhase.UNKNOWN, facts(run_state='running'), StudyState.INTERRUPTED, None),
        (None, JobPhase.UNKNOWN, facts(run_state='crashed'), StudyState.INTERRUPTED, None),
        (
            None,
            JobPhase.UNKNOWN,
            facts(run_state='finished', instances={'train': ('failed', 'TrainingPaused: 23:30')}),
            StudyState.INTERRUPTED,
            'TrainingPaused: 23:30',
        ),
        (None, JobPhase.UNKNOWN, facts(run_state='failed'), StudyState.FAILED, None),
        (
            None,
            JobPhase.UNKNOWN,
            facts(run_state='finished', instances={'transform': 'skipped'}),
            StudyState.FAILED,
            None,
        ),
        ('SLURM COMPLETED', JobPhase.COMPLETED, facts(missing=True), StudyState.MISSING, None),
        (None, JobPhase.UNKNOWN, facts(missing=True), StudyState.MISSING, None),
        (
            None,
            JobPhase.UNKNOWN,
            facts(read_error='JSONDecodeError: x'),
            StudyState.FAILED,
            'JSONDecodeError: x',
        ),
        ('SLURM UNRECOGNIZED', JobPhase.ENDED, facts(), StudyState.MISSING, None),
        (None, JobPhase.ENDED, facts(run_state='failed'), StudyState.FAILED, None),
    ],
)
def test_state_table(
    job_reason: str | None,
    job_phase: JobPhase,
    folder: FolderFacts,
    expected: StudyState,
    reason: str | None,
) -> None:
    assert study_state(job_reason=job_reason, job_phase=job_phase, folder=folder) == (
        expected,
        reason,
    )


def test_study_state_uses_job_phase() -> None:
    assert (
        study_state(job_reason='SLURM TIMEOUT', job_phase=JobPhase.INTERRUPTED, folder=facts())[0]
        is StudyState.INTERRUPTED
    )


def test_ended_job_overrides_a_live_looking_lock() -> None:
    folder = facts(run_state='running', lock_live=True)
    assert (
        study_state(job_reason='SLURM TIMEOUT', job_phase=JobPhase.INTERRUPTED, folder=folder)[0]
        is StudyState.INTERRUPTED
    )


def test_reason_is_the_first_error_line() -> None:
    folder = facts(run_state='failed', instances={'train': ('failed', 'ValueError: bad\nmore')})
    _, reason = study_state(job_reason=None, job_phase=JobPhase.UNKNOWN, folder=folder)
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
        study_directory=tmp_path,
        runs=runs,
        jobs=jobs,
        job_states=make_job_states(states={'7_0': 'COMPLETED'}, phases={'7_0': JobPhase.COMPLETED}),
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
        study_directory=tmp_path,
        runs=[run('g/a/1')],
        jobs=[],
        job_states=make_job_states(states={}, phases={}),
    )[0]
    assert report.state is StudyState.FAILED
    assert report.reason is not None


def test_read_folder_never_raises(tmp_path: Path) -> None:
    (tmp_path / 'status.json').write_text('{')
    read = read_folder(run_directory=tmp_path)
    assert read.snapshot is None
    assert read.read_error is not None
    assert read.lock_live is False


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
        kind=JobKind.RUN,
        submission_id='s2',
        time=utc_now_iso(),
        launcher=LauncherKind.LOCAL,
        job_id='local-1-0',
        job_file=1,
        tasks=(local_task,),
    )
    report = collect_states(
        study_directory=tmp_path,
        runs=[run('g/a/1')],
        jobs=[local_job],
        job_states=make_job_states(states={}, phases={}),
    )[0]
    # the local task id carries the task index, like the SLURM task ids
    assert report.job_id == 'local-1-0_0'
    assert report.attempt == 1


@pytest.mark.parametrize('job_state', ['OUT_OF_MEMORY', 'CANCELLED', 'FAILED'])
def test_failing_job_state_beats_a_stale_running_file(job_state: str) -> None:
    folder = facts(run_state='running', instances={'train': 'running'})
    assert (
        study_state(job_reason=f'SLURM {job_state}', job_phase=JobPhase.FAILED, folder=folder)[0]
        is StudyState.FAILED
    )


def test_time_limit_beats_a_failing_job_state() -> None:
    folder = facts(run_state='running', instances={'train': ('failed', 'TrainingPaused: 1h')})
    assert (
        study_state(job_reason='SLURM CANCELLED', job_phase=JobPhase.FAILED, folder=folder)[0]
        is StudyState.INTERRUPTED
    )


def test_old_training_time_limit_prefix_is_not_interrupted() -> None:
    folder = facts(
        run_state='finished', instances={'train': ('failed', 'TrainingTimeLimitReached: 1h')}
    )
    assert (
        study_state(job_reason=None, job_phase=JobPhase.UNKNOWN, folder=folder)[0]
        is StudyState.FAILED
    )


@pytest.mark.parametrize(
    ('job_state', 'job_phase'),
    [
        ('FAILED', JobPhase.FAILED),
        ('TIMEOUT', JobPhase.INTERRUPTED),
        ('CANCELLED', JobPhase.FAILED),
        (None, JobPhase.UNKNOWN),
    ],
)
def test_job_time_limit_signal_is_interrupted(job_state: str | None, job_phase: JobPhase) -> None:
    folder = facts(
        run_state='crashed',
        instances={'train': ('failed', 'JobTimeLimitSignal: SIGUSR2 before the time limit')},
    )
    state, reason = study_state(
        job_reason=None if job_state is None else f'SLURM {job_state}',
        job_phase=job_phase,
        folder=folder,
    )
    assert state is StudyState.INTERRUPTED
    assert reason == 'JobTimeLimitSignal: SIGUSR2 before the time limit'


def _write_status(folder: Path, run_state: RunState, instance_state: InstanceState) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    write_status(
        path=folder / 'status.json',
        snapshot=RunStatusSnapshot(
            run_state=run_state,
            updated_at=utc_now_iso(),
            instances={'train': InstanceRecord(state=instance_state)},
        ),
    )


def _collect_after_completion(
    tmp_path: Path, *, job_state: str, job_phase: JobPhase, sleep: Callable[[float], None]
) -> list[RunReport]:
    jobs = [job('7', (task(index=0, run_path='g/a/1', attempt=1),))]
    return collect_states(
        study_directory=tmp_path,
        runs=[run('g/a/1')],
        jobs=jobs,
        job_states=make_job_states(states={'7_0': job_state}, phases={'7_0': job_phase}),
        sleep=sleep,
    )


def test_grace_wait_rereads_a_completed_run_until_status_settles(tmp_path: Path) -> None:
    folder = tmp_path / 'g' / 'a' / '1'
    _write_status(folder, RunState.RUNNING, InstanceState.RUNNING)
    calls: list[float] = []

    def sleep(seconds: float) -> None:
        calls.append(seconds)
        if len(calls) == 3:
            _write_status(folder, RunState.FINISHED, InstanceState.DONE)

    reports = _collect_after_completion(
        tmp_path, job_state='COMPLETED', job_phase=JobPhase.COMPLETED, sleep=sleep
    )
    assert reports[0].state is StudyState.DONE
    assert calls == [1, 1, 1]


def test_grace_wait_gives_up_after_15_seconds(tmp_path: Path) -> None:
    _write_status(tmp_path / 'g' / 'a' / '1', RunState.RUNNING, InstanceState.RUNNING)
    calls: list[float] = []
    reports = _collect_after_completion(
        tmp_path, job_state='COMPLETED', job_phase=JobPhase.COMPLETED, sleep=calls.append
    )
    assert len(calls) == 15
    assert reports[0].state is StudyState.INTERRUPTED


def test_grace_wait_skips_other_ended_states(tmp_path: Path) -> None:
    _write_status(tmp_path / 'g' / 'a' / '1', RunState.RUNNING, InstanceState.RUNNING)
    calls: list[float] = []
    _collect_after_completion(
        tmp_path, job_state='TIMEOUT', job_phase=JobPhase.INTERRUPTED, sleep=calls.append
    )
    assert calls == []


def test_grace_wait_for_a_missing_folder_gives_up_after_15_seconds(tmp_path: Path) -> None:
    calls: list[float] = []
    reports = _collect_after_completion(
        tmp_path, job_state='COMPLETED', job_phase=JobPhase.COMPLETED, sleep=calls.append
    )
    assert len(calls) == 15
    assert reports[0].state is StudyState.MISSING


def test_grace_wait_returns_new_facts(tmp_path: Path) -> None:
    folder = tmp_path / 'g' / 'a' / '1'
    _write_status(folder, RunState.RUNNING, InstanceState.RUNNING)
    calls: list[float] = []

    def sleep(seconds: float) -> None:
        calls.append(seconds)
        _write_status(folder, RunState.FINISHED, InstanceState.DONE)

    settled = _grace_wait(
        directories={'g/a/1': folder}, completed_phase=JobPhase.COMPLETED, sleep=sleep
    )
    assert calls == [1]
    assert settled['g/a/1'].snapshot is not None
    assert settled['g/a/1'].snapshot.run_state is RunState.FINISHED


def test_collect_states_local_task_ids_hold_index(tmp_path: Path) -> None:
    record = JobRecord(
        kind=JobKind.RUN,
        submission_id='s1',
        time=utc_now_iso(),
        launcher=LauncherKind.LOCAL,
        job_id='local-1-1',
        job_file=1,
        tasks=(
            task(index=1, run_path='g/a/1', attempt=1),
            task(index=2, run_path='g/a/2', attempt=1),
        ),
    )
    reports = collect_states(
        study_directory=tmp_path,
        runs=[run('g/a/1'), run('g/a/2')],
        jobs=[record],
        job_states=make_job_states(
            states={'local-1-1_1': 'running', 'local-1-1_2': 'pending'},
            phases={'local-1-1_1': JobPhase.RUNNING, 'local-1-1_2': JobPhase.QUEUED},
        ),
    )
    assert [report.job_id for report in reports] == ['local-1-1_1', 'local-1-1_2']
    assert [report.job_state for report in reports] == ['running', 'pending']


def test_report_carries_job_reason_and_flags(tmp_path: Path) -> None:
    report = collect_states(
        study_directory=tmp_path,
        runs=[run('g/a/1')],
        jobs=[job('7', (task(index=0, run_path='g/a/1', attempt=1),))],
        job_states=make_job_states(
            states={'7_0': 'TIMEOUT'},
            phases={'7_0': JobPhase.INTERRUPTED},
            reasons={'7_0': 'SLURM TIMEOUT'},
            flags={'7_0': frozenset({JobFlag.TIME_LIMIT})},
        ),
    )[0]
    assert report.job_reason == 'SLURM TIMEOUT'
    assert report.job_flags == frozenset({JobFlag.TIME_LIMIT})

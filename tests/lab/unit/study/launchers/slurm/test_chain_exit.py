"""Exit status used by chained SLURM jobs after their run command ends."""

from dataclasses import replace
from importlib import import_module
import os
from pathlib import Path, PurePosixPath
import time

import pytest

from tests.lab.unit.study.helpers import context
from tests.support.fake_backend import FakeBackend
from tslab.lab.core.status import InstanceRecord, RunStatusSnapshot, utc_now_iso, write_status
from tslab.lab.enums import InstanceState, LauncherKind, RunState, StudyState
from tslab.lab.jobs.slurm import SlurmError
from tslab.lab.study.commands.resubmit import may_retry
from tslab.lab.study.context import Context, records_context
from tslab.lab.study.definition import StudyRun
from tslab.lab.study.launchers.slurm.chain_exit import chain_exit
from tslab.lab.study.launchers.slurm.records import SlurmJobRecord
from tslab.lab.study.launchers.slurm.requeue import at_requeue_limit, restarted_attempt
from tslab.lab.study.paths import study_folder
from tslab.lab.study.records import (
    append_job,
    JobKind,
    latest_tasks,
    ProgressSnapshot,
    read_jobs,
    TaskRecord,
)
from tslab.lab.study.reporting.state import collect_states, FolderFacts

_STATE_MODULE = import_module('tslab.lab.study.reporting.state')
_CHAIN_EXIT_MODULE = import_module('tslab.lab.study.launchers.slurm.chain_exit')
_REQUEUE_MODULE = import_module('tslab.lab.study.launchers.slurm.requeue')


def _job_record(*, run: StudyRun, job_file: int, index: int) -> SlurmJobRecord:
    task = TaskRecord(
        index=index,
        grid=run.grid,
        run_path=run.run_path,
        command=('python', 'run.py'),
        extra_args=(),
        resources={},
        attempt=1,
        progress=ProgressSnapshot(done_instances=0, newest_checkpoint_mtime=None),
    )
    return SlurmJobRecord(
        kind=JobKind.RUN,
        submission_id='s1',
        time=utc_now_iso(),
        launcher=LauncherKind.SLURM,
        job_id='41',
        job_file=job_file,
        tasks=(task,),
    )


@pytest.mark.parametrize(
    ('run_state', 'instance_state', 'expected'),
    [
        (RunState.FINISHED, InstanceState.DONE, 0),
        (RunState.FINISHED, InstanceState.FAILED, 1),
        (None, None, 1),
    ],
)
def test_chain_exit_is_zero_only_for_done(
    *,
    project: Path,
    fake_backend: FakeBackend,
    run_state: RunState | None,
    instance_state: InstanceState | None,
    expected: int,
) -> None:
    del fake_backend
    ctx = context(project=project, machine_file='cluster.toml')
    run = ctx.runs[0]
    job_file = 3
    index = 17
    job = _job_record(run=run, job_file=job_file, index=index)
    ctx = replace(ctx, jobs=[job])
    if run_state is not None and instance_state is not None:
        run_folder = ctx.study_directory / PurePosixPath(run.run_path)
        write_status(
            path=run_folder / 'status.json',
            snapshot=RunStatusSnapshot(
                run_state=run_state,
                updated_at=utc_now_iso(),
                instances={'train': InstanceRecord(state=instance_state)},
            ),
        )

    assert chain_exit(context=ctx, job_file=job_file, index=index) == expected


def test_chain_exit_rereads_records_when_context_snapshot_is_stale(
    project: Path, fake_backend: FakeBackend
) -> None:
    del fake_backend
    ctx = context(project=project, machine_file='cluster.toml')
    run = ctx.runs[0]
    job = _job_record(run=run, job_file=3, index=17)
    write_status(
        path=ctx.study_directory / run.run_path / 'status.json',
        snapshot=RunStatusSnapshot(
            run_state=RunState.FINISHED,
            updated_at=utc_now_iso(),
            instances={'train': InstanceRecord(state=InstanceState.DONE)},
        ),
    )
    append_job(
        path=study_folder(machine=ctx.machine, study=ctx.study) / 'submissions.jsonl', record=job
    )

    assert chain_exit(context=replace(ctx, jobs=[]), job_file=3, index=17) == 0


def test_chain_exit_waits_for_status_to_reach_shared_storage(
    project: Path, fake_backend: FakeBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    del fake_backend
    ctx = context(project=project, machine_file='cluster.toml')
    run = ctx.runs[0]
    job = _job_record(run=run, job_file=3, index=17)
    ctx = replace(ctx, jobs=[job])
    write_status(
        path=ctx.study_directory / run.run_path / 'status.json',
        snapshot=RunStatusSnapshot(
            run_state=RunState.FINISHED,
            updated_at=utc_now_iso(),
            instances={'train': InstanceRecord(state=InstanceState.DONE)},
        ),
    )
    read_folder = _STATE_MODULE.read_folder
    reads = 0

    def delayed_read(*, run_directory: Path) -> FolderFacts:
        nonlocal reads
        reads += 1
        if reads == 1:
            return FolderFacts(snapshot=None, read_error=None, lock_live=False)
        return read_folder(run_directory=run_directory)

    monkeypatch.setattr(_STATE_MODULE, 'read_folder', delayed_read)
    monkeypatch.setattr(time, 'sleep', lambda _seconds: None)

    assert chain_exit(context=ctx, job_file=3, index=17) == 0
    assert reads >= 2


# ---- chained_retry = "requeue" --------------------------------------------------


def _requeue_context(*, project: Path, max_attempts: int = 3) -> Context:
    cluster = project / 'cluster.toml'
    cluster.write_text(
        cluster.read_text()
        + f'\n[followup]\nchained_retry = "requeue"\nmax_attempts = {max_attempts}\n'
    )
    ctx = context(project=project, machine_file='cluster.toml')
    run = ctx.runs[0]
    job = _job_record(run=run, job_file=3, index=17)
    return replace(ctx, jobs=[job])


def _interrupted(*, ctx: Context, progress: bool) -> None:
    run = ctx.runs[0]
    folder = ctx.study_directory / run.run_path
    write_status(
        path=folder / 'status.json',
        snapshot=RunStatusSnapshot(
            run_state=RunState.RUNNING,
            updated_at=utc_now_iso(),
            instances={
                'train': InstanceRecord(
                    state=InstanceState.DONE if progress else InstanceState.RUNNING
                )
            },
        ),
    )


@pytest.fixture
def requeued(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    monkeypatch.setattr(
        _REQUEUE_MODULE, 'scontrol_requeue', lambda *, task_id: calls.append(task_id)
    )
    return calls


def test_chain_exit_requeues_interrupted_run_with_progress(
    project: Path, fake_backend: FakeBackend, requeued: list[str]
) -> None:
    del fake_backend
    ctx = _requeue_context(project=project)
    _interrupted(ctx=ctx, progress=True)
    assert chain_exit(context=ctx, job_file=3, index=17) == 0
    assert requeued == ['41_17']


def test_chain_exit_does_not_requeue_without_progress(
    project: Path, fake_backend: FakeBackend, requeued: list[str]
) -> None:
    del fake_backend
    ctx = _requeue_context(project=project)
    _interrupted(ctx=ctx, progress=False)
    assert chain_exit(context=ctx, job_file=3, index=17) == 1
    assert requeued == []


def test_chain_exit_does_not_requeue_at_max_attempts(
    project: Path, fake_backend: FakeBackend, requeued: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    del fake_backend
    ctx = _requeue_context(project=project, max_attempts=3)
    _interrupted(ctx=ctx, progress=True)
    (job,) = ctx.jobs
    # why: two earlier requeues by TSLab, each recorded with the next attempt
    requeues = [replace(job, tasks=(replace(job.tasks[0], attempt=attempt),)) for attempt in (2, 3)]
    ctx = replace(ctx, jobs=[job, *requeues])
    monkeypatch.setenv('SLURM_RESTART_COUNT', '2')
    assert chain_exit(context=ctx, job_file=3, index=17) == 1
    assert requeued == []


def test_chain_exit_returns_1_when_scontrol_fails(
    project: Path, fake_backend: FakeBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    del fake_backend

    def fail(*, task_id: str) -> None:
        raise SlurmError(task_id)

    monkeypatch.setattr(_REQUEUE_MODULE, 'scontrol_requeue', fail)
    ctx = _requeue_context(project=project)
    _interrupted(ctx=ctx, progress=True)
    assert chain_exit(context=ctx, job_file=3, index=17) == 1


def test_chain_exit_does_not_requeue_in_other_modes(
    project: Path, fake_backend: FakeBackend, requeued: list[str]
) -> None:
    del fake_backend
    ctx = context(project=project, machine_file='cluster.toml')
    ctx = replace(ctx, jobs=[_job_record(run=ctx.runs[0], job_file=3, index=17)])
    _interrupted(ctx=ctx, progress=True)
    assert chain_exit(context=ctx, job_file=3, index=17) == 1
    assert requeued == []


def test_attempt_counts_restarts() -> None:
    assert restarted_attempt(attempt=1, restart_count=2) == 3
    assert restarted_attempt(attempt=1, restart_count=0) == 1


@pytest.mark.parametrize(
    ('restart_count', 'max_requeues', 'expected'),
    [(0, 2, 'below'), (1, 2, 'at-limit'), (2, None, 'unknown')],
)
def test_requeue_limit_is_a_pure_decision(
    restart_count: int, max_requeues: int | None, expected: str
) -> None:
    assert at_requeue_limit(restart_count=restart_count, max_requeues=max_requeues) is (
        expected == 'at-limit'
    )


def test_chain_exit_logs_when_cluster_config_query_fails(
    project: Path,
    fake_backend: FakeBackend,
    requeued: list[str],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    del fake_backend

    def fail() -> dict[str, str]:
        message = 'cluster config unavailable'
        raise SlurmError(message)

    monkeypatch.setattr(_CHAIN_EXIT_MODULE, 'query_cluster_config', fail)
    ctx = _requeue_context(project=project)
    _interrupted(ctx=ctx, progress=True)

    with caplog.at_level('WARNING'):
        assert chain_exit(context=ctx, job_file=3, index=17) == 0

    assert 'cluster config unavailable' in caplog.text
    assert requeued == ['41_17']


def test_chain_exit_logs_when_requeue_limit_is_invalid(
    project: Path,
    fake_backend: FakeBackend,
    requeued: list[str],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    del fake_backend
    monkeypatch.setattr(
        _CHAIN_EXIT_MODULE, 'query_cluster_config', lambda: {'MaxBatchRequeue': 'invalid'}
    )
    ctx = _requeue_context(project=project)
    _interrupted(ctx=ctx, progress=True)

    with caplog.at_level('WARNING'):
        assert chain_exit(context=ctx, job_file=3, index=17) == 0

    assert 'invalid' in caplog.text
    assert requeued == ['41_17']


def _recorded_requeue_context(*, project: Path, max_attempts: int) -> Context:
    """A requeue context whose submission record is in ``submissions.jsonl``."""
    ctx = _requeue_context(project=project, max_attempts=max_attempts)
    path = study_folder(machine=ctx.machine, study=ctx.study) / 'submissions.jsonl'
    append_job(path=path, record=ctx.jobs[0])
    return replace(ctx, jobs=read_jobs(path=path))


def _reloaded(*, ctx: Context) -> Context:
    """The context with the job records read again, as the next chain-exit sees them."""
    path = study_folder(machine=ctx.machine, study=ctx.study) / 'submissions.jsonl'
    return replace(ctx, jobs=read_jobs(path=path))


def _checkpoint(*, ctx: Context, mtime: float) -> None:
    """Save a ``last.ckpt`` in the run folder with the given modification time."""
    path = ctx.study_directory / ctx.runs[0].run_path / 'train' / 'last.ckpt'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('x')
    os.utime(path, (mtime, mtime))


def _latest_task(*, ctx: Context) -> TaskRecord:
    return latest_tasks(jobs=_reloaded(ctx=ctx).jobs)[ctx.runs[0].run_path][1]


def test_chain_exit_records_the_requeued_attempt_with_fresh_progress(
    project: Path, fake_backend: FakeBackend, requeued: list[str]
) -> None:
    del fake_backend
    ctx = _recorded_requeue_context(project=project, max_attempts=3)
    _interrupted(ctx=ctx, progress=False)
    _checkpoint(ctx=ctx, mtime=1000.0)
    assert chain_exit(context=ctx, job_file=3, index=17) == 0
    assert requeued == ['41_17']
    task = _latest_task(ctx=ctx)
    assert task.attempt == 2
    assert task.progress.newest_checkpoint_mtime == 1000.0
    reports = collect_states(
        study_directory=ctx.study_directory,
        runs=[ctx.runs[0]],
        jobs=_reloaded(ctx=ctx).jobs,
        job_states=ctx.job_states,
    )
    assert reports[0].attempt == 2


def test_requeues_and_followup_stay_within_max_attempts(
    project: Path, fake_backend: FakeBackend, requeued: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    del fake_backend
    ctx = _recorded_requeue_context(project=project, max_attempts=3)
    _interrupted(ctx=ctx, progress=False)
    for restarts, mtime in ((0, 1000.0), (1, 2000.0)):
        monkeypatch.setenv('SLURM_RESTART_COUNT', str(restarts))
        _checkpoint(ctx=ctx, mtime=mtime)
        assert chain_exit(context=_reloaded(ctx=ctx), job_file=3, index=17) == 0
    monkeypatch.setenv('SLURM_RESTART_COUNT', '2')
    _checkpoint(ctx=ctx, mtime=3000.0)
    assert chain_exit(context=_reloaded(ctx=ctx), job_file=3, index=17) == 1
    assert requeued == ['41_17', '41_17']
    reports = collect_states(
        study_directory=ctx.study_directory,
        runs=[ctx.runs[0]],
        jobs=_reloaded(ctx=ctx).jobs,
        job_states=ctx.job_states,
    )
    assert reports[0].attempt == 3
    assert not may_retry(report=replace(reports[0], state=StudyState.INTERRUPTED), context=ctx)


def test_chain_exit_compares_progress_against_the_last_requeue(
    project: Path, fake_backend: FakeBackend, requeued: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    del fake_backend
    ctx = _recorded_requeue_context(project=project, max_attempts=9)
    _interrupted(ctx=ctx, progress=False)
    _checkpoint(ctx=ctx, mtime=1000.0)
    assert chain_exit(context=_reloaded(ctx=ctx), job_file=3, index=17) == 0
    monkeypatch.setenv('SLURM_RESTART_COUNT', '1')
    # no new checkpoint since the requeue: the run is stuck
    assert chain_exit(context=_reloaded(ctx=ctx), job_file=3, index=17) == 1
    assert requeued == ['41_17']


def test_scheduler_requeues_do_not_count_as_attempts(
    project: Path, fake_backend: FakeBackend, requeued: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    del fake_backend
    ctx = _recorded_requeue_context(project=project, max_attempts=3)
    _interrupted(ctx=ctx, progress=True)
    # why: two restarts with no TSLab requeue record were preemptions requeued by SLURM
    monkeypatch.setenv('SLURM_RESTART_COUNT', '2')
    assert chain_exit(context=ctx, job_file=3, index=17) == 0
    task = _latest_task(ctx=ctx)
    assert (task.attempt, task.uncounted_attempts) == (4, 2)


def test_failed_requeue_restores_the_previous_record(
    project: Path, fake_backend: FakeBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    del fake_backend

    def fail(*, task_id: str) -> None:
        raise SlurmError(task_id)

    monkeypatch.setattr(_REQUEUE_MODULE, 'scontrol_requeue', fail)
    ctx = _recorded_requeue_context(project=project, max_attempts=3)
    _interrupted(ctx=ctx, progress=True)
    assert chain_exit(context=ctx, job_file=3, index=17) == 1
    assert _latest_task(ctx=ctx) == ctx.jobs[0].tasks[0]


def test_chain_exit_does_not_requeue_at_max_batch_requeue(
    project: Path, fake_backend: FakeBackend, requeued: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    del fake_backend
    ctx = _requeue_context(project=project, max_attempts=9)
    _interrupted(ctx=ctx, progress=True)
    monkeypatch.setenv('SLURM_RESTART_COUNT', '1')
    monkeypatch.setattr(
        _CHAIN_EXIT_MODULE, 'query_cluster_config', lambda: {'MaxBatchRequeue': '2'}
    )
    assert chain_exit(context=ctx, job_file=3, index=17) == 1
    assert requeued == []


def test_chain_exit_reads_records_only_without_scheduler_or_other_runs(
    project: Path, fake_backend: FakeBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = context(project=project, machine_file='cluster.toml')
    run = ctx.runs[0]
    job = _job_record(run=run, job_file=3, index=17)
    append_job(
        path=study_folder(machine=ctx.machine, study=ctx.study) / 'submissions.jsonl', record=job
    )
    write_status(
        path=ctx.study_directory / run.run_path / 'status.json',
        snapshot=RunStatusSnapshot(
            run_state=RunState.FINISHED,
            updated_at=utc_now_iso(),
            instances={'train': InstanceRecord(state=InstanceState.DONE)},
        ),
    )
    before = len(fake_backend.calls('job_states'))
    read: list[Path] = []
    read_folder = _STATE_MODULE.read_folder

    def counted_read(*, run_directory: Path) -> FolderFacts:
        read.append(run_directory)
        return read_folder(run_directory=run_directory)

    monkeypatch.setattr(_STATE_MODULE, 'read_folder', counted_read)

    light = records_context(
        reference='toy_studies:study',
        machine_path=project / 'cluster.toml',
        project_root=project,
        environment=os.environ,
    )

    assert len(fake_backend.calls('job_states')) == before
    assert light.reports == []
    assert light.jobs == [job]
    assert chain_exit(context=light, job_file=3, index=17) == 0
    assert read == [ctx.study_directory / run.run_path]

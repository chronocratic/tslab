"""``tslab study resubmit``: the interrupted, missing and failed runs again."""

import dataclasses
import importlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import cast, TYPE_CHECKING

import pytest

from tests.lab.unit.study.helpers import (
    chained_context,
    chained_study,
    context,
    finish_run,
    jobs_of,
    no_sleep,
    record_local_job,
    study_directory,
    submit_study,
    write_launch,
    write_run_status,
)
from tests.support.fake_backend import FakeBackend
from tslab.lab.core.meta import hostname
from tslab.lab.core.paths import LOCK_FILE
from tslab.lab.core.status import InstanceRecord, RunStatusSnapshot, utc_now_iso, write_status
from tslab.lab.enums import AfterCondition, InstanceState, JobFlag, JobPhase, RunState, StudyState
from tslab.lab.study.code_snapshot import CodeSnapshot, CodeSnapshotError
from tslab.lab.study.commands import resubmit, submit, submit_runs
from tslab.lab.study.commands.resubmit import _remove_ended_locks
from tslab.lab.study.context import Context, load_context, refresh_context, Selection, study_reports
from tslab.lab.study.launchers.common.types import LaunchError
from tslab.lab.study.launchers.slurm.chain_exit import chain_exit
from tslab.lab.study.records import JobRecord, read_jobs
from tslab.lab.study.reporting.state import RunReport

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tslab.lab.study.plan import JobGroup


def running_run_with_lock(*, project: Path, model: str, seed: int) -> None:
    """A run folder that says running and still holds a live foreign lock."""
    folder = project / 'runs' / 's' / f'g/{model}/{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    write_status(
        path=folder / 'status.json',
        snapshot=RunStatusSnapshot(
            run_state=RunState.RUNNING,
            updated_at=utc_now_iso(),
            instances={'train': InstanceRecord(state=InstanceState.RUNNING)},
        ),
    )
    (folder / '.lock').write_text(json.dumps({'pid': 99999, 'host': 'otherhost'}))


def enable_retry(*, project: Path, text: str = 'retry = "resubmit"\n') -> None:
    """Add a ``[followup]`` table with the given settings to the SLURM profile."""
    cluster = project / 'cluster.toml'
    cluster.write_text(cluster.read_text() + '\n[followup]\n' + text)


def test_default_states_are_interrupted_and_missing(
    project: Path, fake_backend: FakeBackend
) -> None:
    submit_study(project=project)
    fake_backend.set_states(
        phases={
            'job-0_0': JobPhase.COMPLETED,
            'job-0_1': JobPhase.FAILED,
            'job-0_2': JobPhase.INTERRUPTED,
            'job-0_3': JobPhase.COMPLETED,
        }
    )
    finish_run(project=project, model='small', seed=1)
    code = resubmit(context=context(project=project, machine_file='cluster.toml'), sleep=no_sleep)
    assert code == 0
    jobs = jobs_of(project=project)
    assert jobs[1].job_id == 'job-1'
    assert [t.run_path.as_posix() for t in jobs[1].tasks] == ['g/large/1', 'g/large/2']
    assert all(t.attempt == 2 for t in jobs[1].tasks)


def test_failed_must_be_asked_for(project: Path, fake_backend: FakeBackend) -> None:
    submit_study(project=project)
    fake_backend.set_states(phases={'job-0_1': JobPhase.FAILED})
    code = resubmit(
        context=context(project=project, machine_file='cluster.toml'),
        states=(StudyState.FAILED,),
        sleep=no_sleep,
    )
    assert code == 0
    jobs = jobs_of(project=project)
    assert [t.run_path.as_posix() for t in jobs[1].tasks] == ['g/small/2']


def test_resubmit_never_touches_live_runs(project: Path, fake_backend: FakeBackend) -> None:
    submit_study(project=project)
    fake_backend.set_states(phases={'job-0_0': JobPhase.RUNNING})
    all_states = tuple(StudyState)
    code = resubmit(
        context=context(project=project, machine_file='cluster.toml'),
        states=all_states,
        sleep=no_sleep,
    )
    assert code == 0
    jobs = jobs_of(project=project)
    assert all(t.run_path != PurePosixPath('g/small/1') for t in jobs[1].tasks)
    assert fake_backend.calls('cancel') == []


def test_cancel_live_cancels_then_resubmits_later(project: Path, fake_backend: FakeBackend) -> None:
    live_cancel_setup(project=project, fake_backend=fake_backend)
    code = resubmit(
        context=context(project=project, machine_file='cluster.toml'),
        cancel_live=True,
        sleep=no_sleep,
    )
    assert code == 0
    assert [(call['dry_run'], call['signal']) for call in fake_backend.calls('cancel')] == [
        (False, None)
    ]
    jobs = jobs_of(project=project)
    assert [t.run_path.as_posix() for t in jobs[1].tasks] == ['g/small/1']
    assert fake_backend.calls('launch')[-1]['start_delay_seconds'] == 660


def test_resubmit_checks_local_launch_before_submission_lock(
    project: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    ctx = context(project=project, machine_file='local.toml')
    assert any(report.state is StudyState.MISSING for report in ctx.reports)
    write_launch(
        folder=study_directory(project=project) / '.study',
        pid=os.getpid(),
        host=hostname(),
        tasks={},
    )
    submit_module = importlib.import_module('tslab.lab.study.commands.submit')
    monkeypatch.setattr(
        submit_module,
        'submission_lock',
        lambda **_kwargs: pytest.fail('resubmit reached submission_lock'),
    )

    with caplog.at_level('ERROR'):
        code = resubmit(context=ctx, sleep=no_sleep)

    assert code == 1
    assert 'a local launch is running' in caplog.text


def test_ended_job_lock_is_removed_before_resubmitting(
    project: Path, fake_backend: FakeBackend, caplog: pytest.LogCaptureFixture
) -> None:
    submit_study(project=project)
    fake_backend.set_states(phases={'job-0_3': JobPhase.INTERRUPTED})
    running_run_with_lock(project=project, model='large', seed=2)
    with caplog.at_level('INFO'):
        code = resubmit(
            context=context(project=project, machine_file='cluster.toml'), sleep=no_sleep
        )
    assert code == 0
    lock = project / 'runs' / 's' / 'g/large/2' / '.lock'
    assert not lock.exists()
    assert 'removed the lock of g/large/2: its job job-0_3 ended (interrupted)' in caplog.text
    jobs = jobs_of(project=project)
    assert 'g/large/2' in [t.run_path.as_posix() for t in jobs[1].tasks]


def test_pass_through_arguments_reach_jobs_and_retries(
    project: Path, fake_backend: FakeBackend, tmp_path: Path
) -> None:
    track = tmp_path / 'track.log'
    submit_study(project=project, extra_args=('--tracking-log', str(track)))
    first = jobs_of(project=project)[0]
    assert first.tasks[0].extra_args == ('--tracking-log', str(track))
    assert first.tasks[0].command[-2:] == ('--tracking-log', str(track))
    code = resubmit(
        context=context(project=project, machine_file='cluster.toml'),
        extra_args=None,
        sleep=no_sleep,
    )
    assert code == 0
    kept = jobs_of(project=project)[1].tasks[0]
    assert kept.extra_args == ('--tracking-log', str(track))
    other = tmp_path / 'other.log'
    code = resubmit(
        context=context(project=project, machine_file='cluster.toml'),
        extra_args=('--tracking-log', str(other)),
        sleep=no_sleep,
    )
    assert code == 0
    replaced = jobs_of(project=project)[2].tasks[0]
    assert replaced.extra_args == ('--tracking-log', str(other))
    assert replaced.command[-2:] == ('--tracking-log', str(other))


def live_cancel_setup(*, project: Path, fake_backend: FakeBackend) -> None:
    """Task 0 queued and tasks 1 to 3 done, so only one run is live."""
    submit_study(project=project)
    fake_backend.set_states(
        phases={
            'job-0_0': JobPhase.QUEUED,
            'job-0_1': JobPhase.COMPLETED,
            'job-0_2': JobPhase.COMPLETED,
            'job-0_3': JobPhase.COMPLETED,
        }
    )
    for model, seed in (('small', 2), ('large', 1), ('large', 2)):
        finish_run(project=project, model=model, seed=seed)
    # why: the live run of job-0 is what a cancel can reach, as on a real scheduler
    fake_backend.cancellable_answer = ['job-0_0']


def test_dry_run_with_cancel_live_cancels_and_unlocks_nothing(
    project: Path, fake_backend: FakeBackend, caplog: pytest.LogCaptureFixture
) -> None:
    live_cancel_setup(project=project, fake_backend=fake_backend)
    fake_backend.set_states(
        phases={
            'job-0_0': JobPhase.QUEUED,
            'job-0_1': JobPhase.COMPLETED,
            'job-0_2': JobPhase.COMPLETED,
            'job-0_3': JobPhase.INTERRUPTED,
        }
    )
    running_run_with_lock(project=project, model='large', seed=2)
    with caplog.at_level('INFO'):
        code = resubmit(
            context=context(project=project, machine_file='cluster.toml'),
            cancel_live=True,
            dry_run=True,
            sleep=no_sleep,
        )
    assert code == 0
    assert fake_backend.calls('cancel') == []
    assert (project / 'runs' / 's' / 'g/large/2' / '.lock').exists()
    assert 'would cancel job-0_0' in caplog.text
    assert len(jobs_of(project=project)) == 1


def test_failed_validation_cancels_nothing(project: Path, fake_backend: FakeBackend) -> None:
    live_cancel_setup(project=project, fake_backend=fake_backend)
    # why: a pass-through flag the config does not have makes every run fail validation
    code = resubmit(
        context=context(project=project, machine_file='cluster.toml'),
        cancel_live=True,
        extra_args=('--no-such-flag',),
        sleep=no_sleep,
    )
    assert code == 1
    assert fake_backend.calls('cancel') == []
    assert len(jobs_of(project=project)) == 1


def test_cancel_failure_exits_one_and_submits_nothing(
    project: Path, fake_backend: FakeBackend
) -> None:
    live_cancel_setup(project=project, fake_backend=fake_backend)
    fake_backend.errors['cancel'] = LaunchError('Invalid job id')
    code = resubmit(
        context=context(project=project, machine_file='cluster.toml'),
        cancel_live=True,
        sleep=no_sleep,
    )
    assert code == 1
    assert len(fake_backend.calls('launch')) == 1
    assert len(jobs_of(project=project)) == 1


def test_unknown_slurm_states_refuse_to_submit(
    project: Path, fake_backend: FakeBackend, caplog: pytest.LogCaptureFixture
) -> None:
    submit_study(project=project)
    ctx = context(project=project, machine_file='cluster.toml')
    unknown = dataclasses.replace(ctx, job_states=dataclasses.replace(ctx.job_states, known=False))
    with caplog.at_level('ERROR'):
        code = resubmit(context=unknown, states=tuple(StudyState), sleep=no_sleep)
    assert code == 1
    assert len(fake_backend.calls('launch')) == 1
    assert 'not submitting' in caplog.text


def test_auto_resubmit_retries_paused_run_with_progress(
    project: Path, fake_backend: FakeBackend
) -> None:
    submit_study(project=project)
    # g/small/1 paused by job_time_share; it saved a checkpoint before the pause
    write_run_status(
        project=project,
        model='small',
        seed=1,
        run_state=RunState.FAILED,
        instance=InstanceState.FAILED,
        error='TrainingPaused: job time share reached',
    )
    (project / 'runs' / 's' / 'g/small/1' / 'train').mkdir()
    (project / 'runs' / 's' / 'g/small/1' / 'train' / 'last.ckpt').write_text('ckpt')
    fake_backend.set_states(phases={'job-0_0': JobPhase.COMPLETED})
    enable_retry(project=project)
    code = resubmit(
        context=context(project=project, machine_file='cluster.toml'),
        auto=True,
        dry_run=True,
        sleep=no_sleep,
    )
    assert code == 0
    # the dry-run plan holds only the paused run, one attempt further on
    call = fake_backend.calls('launch')[-1]
    assert call['dry_run'] is True
    (group,) = cast('Sequence[JobGroup]', call['groups'])
    (task,) = group.tasks
    assert task.index == 0
    assert task.attempt == 2
    command = list(task.command)
    assert command[command.index('--run-path') + 1] == 'g/small/1'
    assert len(jobs_of(project=project)) == 1


def _local_retry_context(*, project: Path, max_attempts: int) -> Context:
    """The local context of the run g/small/1, with the automatic retry on."""
    local = project / 'local.toml'
    local.write_text(
        local.read_text() + f'\n[followup]\nretry = "resubmit"\nmax_attempts = {max_attempts}\n'
    )
    return load_context(
        reference='toy_studies:study',
        machine_path=local,
        selection=Selection(only=('model=small', 'seed=1')),
        project_root=Path.cwd(),
        local_cpus=1,
        environment=os.environ,
    )


def _local_attempts(*, project: Path) -> int:
    return int((project / 'runs' / 's' / 'g/small/1' / 'attempts').read_text())


def test_local_retry_finishes_after_three_attempts(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('TOY_SUCCEED_ON', '3')
    ctx = _local_retry_context(project=project, max_attempts=3)
    submit(context=ctx, include_done=False, dry_run=False, extra_args=(), sleep=no_sleep)
    assert _local_attempts(project=project) == 3
    report = refresh_context(context=ctx).reports[0]
    assert report.state is StudyState.DONE
    assert len(jobs_of(project=project)) == 3


def test_local_retry_stops_at_max_attempts(
    project: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv('TOY_SUCCEED_ON', '3')
    ctx = _local_retry_context(project=project, max_attempts=2)
    with caplog.at_level('INFO'):
        submit(context=ctx, include_done=False, dry_run=False, extra_args=(), sleep=no_sleep)
    assert _local_attempts(project=project) == 2
    assert refresh_context(context=ctx).reports[0].state is StudyState.INTERRUPTED
    assert 'gave up' in caplog.text


def test_local_no_retry_after_signal(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _local_retry_context(project=project, max_attempts=3)
    calls: list[object] = []
    # the door re-exports `submit` (the function), shadowing the submodule; reach it by name
    submit_module = importlib.import_module('tslab.lab.study.commands.submit')
    monkeypatch.setattr(submit_module, 'submit_runs', lambda **_: (143, []))
    monkeypatch.setattr('tslab.lab.study.commands.resubmit', lambda **kwargs: calls.append(kwargs))
    code = submit(context=ctx, include_done=False, dry_run=False, extra_args=(), sleep=no_sleep)
    assert code == 143
    assert calls == []


def _ended_run_with_lock(*, project: Path, host: str) -> tuple[RunReport, Path]:
    """A local run whose job ended while its folder holds a live lock of ``host``."""
    record_local_job(project=project)
    ctx = context(project=project, machine_file='local.toml')
    folder = study_directory(project=project) / ctx.runs[0].run_path
    folder.mkdir(parents=True)
    write_status(
        path=folder / 'status.json',
        snapshot=RunStatusSnapshot(
            run_state=RunState.RUNNING,
            updated_at=utc_now_iso(),
            instances={'train': InstanceRecord(state=InstanceState.RUNNING)},
        ),
    )
    lock = folder / LOCK_FILE
    lock.write_text(f'{{"pid": {os.getpid()}, "host": "{host}", "nonce": "n"}}', encoding='utf-8')
    report = context(project=project, machine_file='local.toml').reports[0]
    assert report.job_phase is JobPhase.ENDED
    assert report.folder.lock_live
    return report, lock


def test_a_live_lock_of_this_host_is_kept(project: Path) -> None:
    report, lock = _ended_run_with_lock(project=project, host=hostname())
    _remove_ended_locks(
        reports=[report], study_directory=study_directory(project=project), knows_other_hosts=False
    )
    assert lock.exists()


def test_a_live_lock_of_another_host_is_kept_on_a_local_machine(
    project: Path, caplog: pytest.LogCaptureFixture
) -> None:
    report, lock = _ended_run_with_lock(project=project, host='other-host')
    with caplog.at_level('INFO'):
        _remove_ended_locks(
            reports=[report],
            study_directory=study_directory(project=project),
            knows_other_hosts=False,
        )
    assert lock.exists()
    assert f'kept the lock of {report.run.run_path}: it is live on host other-host' in caplog.text


def test_a_live_lock_of_another_host_is_removed_when_the_backend_knows_other_hosts(
    project: Path,
) -> None:
    report, lock = _ended_run_with_lock(project=project, host='other-host')
    _remove_ended_locks(
        reports=[report], study_directory=study_directory(project=project), knows_other_hosts=True
    )
    assert not lock.exists()


def _snapshot(*, folder: Path, digest: str, with_src: bool = True) -> CodeSnapshot:
    """One recorded code snapshot under ``folder``; ``src/`` exists unless told not to."""
    if with_src:
        (folder / 'src').mkdir(parents=True)
        (folder / 'src' / 'mod.py').write_text('X = 1\n', encoding='utf-8')
    return CodeSnapshot(
        folder=folder, hash=digest, commit='c' * 40, dirty=False, untracked_copied=False
    )


def _end_all(*, project: Path, fake_backend: FakeBackend, phase: JobPhase) -> None:
    """Every recorded task ends in ``phase``, so the whole study is resubmittable."""
    jobs = jobs_of(project=project)
    phases = {f'{job.job_id}_{task.index}': phase for job in jobs for task in job.tasks}
    fake_backend.set_states(phases=phases)


def test_resubmit_reuses_recorded_code_snapshot(
    project: Path, tmp_path: Path, fake_backend: FakeBackend
) -> None:
    snapshot = _snapshot(folder=tmp_path / 'snap', digest='a' * 16)
    code = submit(
        context=context(project=project, machine_file='cluster.toml'),
        include_done=False,
        dry_run=False,
        extra_args=(),
        code_snapshot=snapshot,
        sleep=no_sleep,
    )
    assert code == 0
    _end_all(project=project, fake_backend=fake_backend, phase=JobPhase.INTERRUPTED)
    code = resubmit(context=context(project=project, machine_file='cluster.toml'), sleep=no_sleep)
    assert code == 0
    latest = jobs_of(project=project)[-1].code_snapshot
    assert latest is not None
    assert latest.folder == snapshot.folder
    assert latest.hash == snapshot.hash


def test_resubmit_mixed_runs_keep_their_own(
    project: Path, tmp_path: Path, fake_backend: FakeBackend
) -> None:
    first = _snapshot(folder=tmp_path / 'snap-a', digest='a' * 16)
    second = _snapshot(folder=tmp_path / 'snap-b', digest='b' * 16)
    ctx = context(project=project, machine_file='cluster.toml')
    code, _ = submit_runs(
        context=ctx,
        reports=ctx.reports,
        dry_run=False,
        extra_args={report.run.run_path: () for report in ctx.reports},
        code_snapshots={
            PurePosixPath('g/small/1'): first,
            PurePosixPath('g/small/2'): second,
            PurePosixPath('g/large/1'): first,
            PurePosixPath('g/large/2'): None,
        },
        sleep=no_sleep,
    )
    assert code == 0
    _end_all(project=project, fake_backend=fake_backend, phase=JobPhase.INTERRUPTED)
    code = resubmit(context=context(project=project, machine_file='cluster.toml'), sleep=no_sleep)
    assert code == 0
    new = jobs_of(project=project)[3:]
    assert len(new) == 3
    by_path = {
        task.run_path: job.code_snapshot.folder if job.code_snapshot is not None else None
        for job in new
        for task in job.tasks
    }
    assert by_path == {
        PurePosixPath('g/small/1'): first.folder,
        PurePosixPath('g/small/2'): second.folder,
        PurePosixPath('g/large/1'): first.folder,
        PurePosixPath('g/large/2'): None,
    }


def test_resubmit_new_code_snapshot_replaces(
    project: Path, tmp_path: Path, fake_backend: FakeBackend
) -> None:
    recorded = _snapshot(folder=tmp_path / 'snap-old', digest='o' * 16)
    code = submit(
        context=context(project=project, machine_file='cluster.toml'),
        include_done=False,
        dry_run=False,
        extra_args=(),
        code_snapshot=recorded,
        sleep=no_sleep,
    )
    assert code == 0
    _end_all(project=project, fake_backend=fake_backend, phase=JobPhase.INTERRUPTED)
    fresh = _snapshot(folder=tmp_path / 'snap-new', digest='n' * 16)
    code = resubmit(
        context=context(project=project, machine_file='cluster.toml'),
        new_code_snapshot=fresh,
        sleep=no_sleep,
    )
    assert code == 0
    latest = jobs_of(project=project)[-1].code_snapshot
    assert latest is not None
    assert latest.folder == fresh.folder


def test_resubmit_drop_code_snapshot_runs_live(
    project: Path, tmp_path: Path, fake_backend: FakeBackend
) -> None:
    recorded = _snapshot(folder=tmp_path / 'snap', digest='a' * 16)
    code = submit(
        context=context(project=project, machine_file='cluster.toml'),
        include_done=False,
        dry_run=False,
        extra_args=(),
        code_snapshot=recorded,
        sleep=no_sleep,
    )
    assert code == 0
    _end_all(project=project, fake_backend=fake_backend, phase=JobPhase.INTERRUPTED)
    code = resubmit(
        context=context(project=project, machine_file='cluster.toml'),
        drop_code_snapshot=True,
        sleep=no_sleep,
    )
    assert code == 0
    assert jobs_of(project=project)[-1].code_snapshot is None


def test_resubmit_missing_recorded_folder_is_an_error(
    project: Path, tmp_path: Path, fake_backend: FakeBackend
) -> None:
    gone = _snapshot(folder=tmp_path / 'snap-gone', digest='e' * 16, with_src=False)
    code = submit(
        context=context(project=project, machine_file='cluster.toml'),
        include_done=False,
        dry_run=False,
        extra_args=(),
        code_snapshot=gone,
        sleep=no_sleep,
    )
    assert code == 0
    _end_all(project=project, fake_backend=fake_backend, phase=JobPhase.INTERRUPTED)
    with pytest.raises(CodeSnapshotError) as raised:
        resubmit(context=context(project=project, machine_file='cluster.toml'), sleep=no_sleep)
    message = str(raised.value)
    assert str(gone.folder) in message
    assert '--new-code-snapshot' in message
    assert len(jobs_of(project=project)) == 1


def test_resubmit_failed_warns_on_reuse(
    project: Path, tmp_path: Path, fake_backend: FakeBackend, caplog: pytest.LogCaptureFixture
) -> None:
    first = _snapshot(folder=tmp_path / 'snap-a', digest='a' * 16)
    second = _snapshot(folder=tmp_path / 'snap-b', digest='b' * 16)
    ctx = context(project=project, machine_file='cluster.toml')
    code, _ = submit_runs(
        context=ctx,
        reports=ctx.reports,
        dry_run=False,
        extra_args={report.run.run_path: () for report in ctx.reports},
        code_snapshots={
            PurePosixPath('g/small/1'): first,
            PurePosixPath('g/small/2'): first,
            PurePosixPath('g/large/1'): second,
            PurePosixPath('g/large/2'): second,
        },
        sleep=no_sleep,
    )
    assert code == 0
    original = jobs_of(project=project)
    first_time = next(
        job.time
        for job in original
        if job.code_snapshot is not None and job.code_snapshot.hash == first.hash
    )
    second_time = next(
        job.time
        for job in original
        if job.code_snapshot is not None and job.code_snapshot.hash == second.hash
    )
    _end_all(project=project, fake_backend=fake_backend, phase=JobPhase.FAILED)
    with caplog.at_level('WARNING'):
        code = resubmit(
            context=context(project=project, machine_file='cluster.toml'),
            states=(StudyState.FAILED,),
            sleep=no_sleep,
        )
    assert code == 0
    assert (
        f'reusing the code snapshot from {first_time}; '
        'pass --new-code-snapshot if you fixed the code'
    ) in caplog.text
    assert (
        f'reusing the code snapshot from {second_time}; '
        'pass --new-code-snapshot if you fixed the code'
    ) in caplog.text
    assert caplog.text.count('reusing the code snapshot from') == 2


def test_resubmit_failed_with_new_code_snapshot_no_warning(
    project: Path, tmp_path: Path, fake_backend: FakeBackend, caplog: pytest.LogCaptureFixture
) -> None:
    recorded = _snapshot(folder=tmp_path / 'snap', digest='a' * 16)
    code = submit(
        context=context(project=project, machine_file='cluster.toml'),
        include_done=False,
        dry_run=False,
        extra_args=(),
        code_snapshot=recorded,
        sleep=no_sleep,
    )
    assert code == 0
    _end_all(project=project, fake_backend=fake_backend, phase=JobPhase.FAILED)
    fresh = _snapshot(folder=tmp_path / 'snap-new', digest='n' * 16)
    with caplog.at_level('WARNING'):
        code = resubmit(
            context=context(project=project, machine_file='cluster.toml'),
            states=(StudyState.FAILED,),
            new_code_snapshot=fresh,
            sleep=no_sleep,
        )
    assert code == 0
    assert 'reusing the code snapshot from' not in caplog.text


def test_auto_resubmit_reuses_recorded_code_snapshot(
    project: Path, tmp_path: Path, fake_backend: FakeBackend
) -> None:
    snapshot = _snapshot(folder=tmp_path / 'snap', digest='a' * 16)
    code = submit(
        context=context(project=project, machine_file='cluster.toml'),
        include_done=False,
        dry_run=False,
        extra_args=(),
        code_snapshot=snapshot,
        sleep=no_sleep,
    )
    assert code == 0
    # g/small/1 paused by job_time_share; it saved a checkpoint before the pause
    write_run_status(
        project=project,
        model='small',
        seed=1,
        run_state=RunState.FAILED,
        instance=InstanceState.FAILED,
        error='TrainingPaused: job time share reached',
    )
    (project / 'runs' / 's' / 'g/small/1' / 'train').mkdir()
    (project / 'runs' / 's' / 'g/small/1' / 'train' / 'last.ckpt').write_text('ckpt')
    fake_backend.set_states(phases={'job-0_0': JobPhase.COMPLETED})
    enable_retry(project=project)
    code = resubmit(
        context=context(project=project, machine_file='cluster.toml'), auto=True, sleep=no_sleep
    )
    assert code == 0
    latest = jobs_of(project=project)[-1]
    assert [task.run_path.as_posix() for task in latest.tasks] == ['g/small/1']
    assert latest.code_snapshot is not None
    assert latest.code_snapshot.folder == snapshot.folder


# ---- chained grids -------------------------------------------------------------


def _chain_submitted(*, project: Path, fake_backend: FakeBackend) -> Context:
    """The chained study submitted once on the SLURM profile: job-0 pretrain, job-1/2 probe."""
    ctx = chained_context(project=project, study=chained_study(), machine_file='cluster.toml')
    assert (
        submit(context=ctx, include_done=False, dry_run=False, extra_args=(), sleep=no_sleep) == 0
    )
    assert [
        cast('Sequence[JobGroup]', call['groups'])[0].grid for call in fake_backend.calls('launch')
    ] == ['pretrain']
    return ctx


def _chain_states(
    *,
    project: Path,
    fake_backend: FakeBackend,
    phases: dict[str, JobPhase],
    flags: dict[str, frozenset[JobFlag]] | None = None,
    grids: tuple[str, ...] = (),
) -> Context:
    fake_backend.set_states(phases=phases, flags=flags or {})
    return chained_context(
        project=project, study=chained_study(), machine_file='cluster.toml', grids=grids
    )


def _launched_runs(*, fake_backend: FakeBackend) -> list[str]:
    (call,) = fake_backend.calls('launch')[-1:]
    return [
        task.resolved.run.run_path.as_posix()
        for group in cast('Sequence[JobGroup]', call['groups'])
        for task in group.tasks
    ]


def test_default_resubmit_takes_dependency_failed(project: Path, fake_backend: FakeBackend) -> None:
    _chain_submitted(project=project, fake_backend=fake_backend)
    ctx = _chain_states(
        project=project,
        fake_backend=fake_backend,
        phases={
            'job-0_0': JobPhase.INTERRUPTED,
            'job-0_1': JobPhase.INTERRUPTED,
            'job-1_0': JobPhase.FAILED,
            'job-1_1': JobPhase.FAILED,
        },
        flags={
            'job-1_0': frozenset({JobFlag.DEPENDENCY_NEVER}),
            'job-1_1': frozenset({JobFlag.DEPENDENCY_NEVER}),
        },
    )
    assert {r.state for r in ctx.reports if r.run.grid == 'probe'} >= {StudyState.DEPENDENCY_FAILED}
    assert resubmit(context=ctx, sleep=no_sleep) == 0
    runs = _launched_runs(fake_backend=fake_backend)
    assert 'pretrain/a' in runs
    assert any(run.startswith('probe/a/') for run in runs)


def test_resubmit_upstream_pulls_in_waiting_downstream(
    project: Path, fake_backend: FakeBackend
) -> None:
    _chain_submitted(project=project, fake_backend=fake_backend)
    ctx = _chain_states(
        project=project,
        fake_backend=fake_backend,
        phases={
            'job-0_0': JobPhase.INTERRUPTED,
            'job-0_1': JobPhase.COMPLETED,
            'job-1_0': JobPhase.QUEUED,
            'job-1_1': JobPhase.QUEUED,
            'job-2_2': JobPhase.QUEUED,
            'job-2_3': JobPhase.QUEUED,
        },
        flags={
            task: frozenset({JobFlag.DEPENDENCY_PENDING})
            for task in ('job-1_0', 'job-1_1', 'job-2_2', 'job-2_3')
        },
        grids=('pretrain',),
    )
    fake_backend.cancellable_answer = ['job-1']
    assert resubmit(context=ctx, sleep=no_sleep) == 0
    (cancel_call,) = fake_backend.calls('cancel')
    cancelled = cast('Context', cancel_call['context']).reports
    assert {r.run.grid for r in cancelled} == {'probe'}
    runs = _launched_runs(fake_backend=fake_backend)
    assert 'pretrain/a' in runs
    assert any(run.startswith('probe/a/') for run in runs)


def test_auto_ignores_chained_runs_when_chained_retry_manual(
    project: Path, fake_backend: FakeBackend
) -> None:
    _chain_submitted(project=project, fake_backend=fake_backend)
    enable_retry(project=project, text='retry = "resubmit"\n')
    ctx = _chain_states(
        project=project,
        fake_backend=fake_backend,
        phases={'job-0_0': JobPhase.INTERRUPTED, 'job-0_1': JobPhase.INTERRUPTED},
    )
    (project / 'runs' / 'chain' / 'pretrain/a' / 'train').mkdir(parents=True, exist_ok=True)
    (project / 'runs' / 'chain' / 'pretrain/a' / 'train' / 'last.ckpt').write_text('x')
    before = len(fake_backend.calls('launch'))
    assert resubmit(context=ctx, auto=True, sleep=no_sleep) == 0
    assert len(fake_backend.calls('launch')) == before


def test_auto_ignores_unchained_runs_when_retry_manual(
    project: Path, fake_backend: FakeBackend
) -> None:
    submit_study(project=project)
    write_run_status(
        project=project,
        model='small',
        seed=1,
        run_state=RunState.FAILED,
        instance=InstanceState.FAILED,
        error='TrainingPaused: job time share reached',
    )
    (project / 'runs' / 's' / 'g/small/1' / 'train').mkdir()
    (project / 'runs' / 's' / 'g/small/1' / 'train' / 'last.ckpt').write_text('ckpt')
    fake_backend.set_states(phases={'job-0_0': JobPhase.COMPLETED})
    enable_retry(project=project, text='chained_retry = "resubmit"\n')
    before = len(fake_backend.calls('launch'))
    assert (
        resubmit(
            context=context(project=project, machine_file='cluster.toml'), auto=True, sleep=no_sleep
        )
        == 0
    )
    assert len(fake_backend.calls('launch')) == before


def test_auto_resubmits_time_limited_upstream_and_its_probe(
    project: Path, fake_backend: FakeBackend
) -> None:
    _chain_submitted(project=project, fake_backend=fake_backend)
    enable_retry(project=project, text='chained_retry = "resubmit"\n')
    ctx = _chain_states(
        project=project,
        fake_backend=fake_backend,
        phases={
            'job-0_0': JobPhase.INTERRUPTED,
            'job-0_1': JobPhase.COMPLETED,
            'job-1_0': JobPhase.QUEUED,
            'job-1_1': JobPhase.QUEUED,
        },
        flags={
            'job-1_0': frozenset({JobFlag.DEPENDENCY_PENDING}),
            'job-1_1': frozenset({JobFlag.DEPENDENCY_PENDING}),
        },
        grids=('pretrain',),
    )
    folder = project / 'runs' / 'chain' / 'pretrain/a'
    (folder / 'train').mkdir(parents=True, exist_ok=True)
    (folder / 'train' / 'last.ckpt').write_text('x')
    fake_backend.cancellable_answer = ['job-1']
    assert resubmit(context=ctx, auto=True, sleep=no_sleep) == 0
    runs = _launched_runs(fake_backend=fake_backend)
    assert 'pretrain/a' in runs
    assert any(run.startswith('probe/a/') for run in runs)


def test_auto_does_not_resubmit_probe_of_failed_upstream(
    project: Path, fake_backend: FakeBackend
) -> None:
    _chain_submitted(project=project, fake_backend=fake_backend)
    enable_retry(project=project, text='chained_retry = "resubmit"\n')
    ctx = _chain_states(
        project=project,
        fake_backend=fake_backend,
        phases={
            'job-0_0': JobPhase.FAILED,
            'job-0_1': JobPhase.FAILED,
            'job-1_0': JobPhase.FAILED,
            'job-1_1': JobPhase.FAILED,
        },
        flags={
            'job-1_0': frozenset({JobFlag.DEPENDENCY_NEVER}),
            'job-1_1': frozenset({JobFlag.DEPENDENCY_NEVER}),
        },
    )
    before = len(fake_backend.calls('launch'))
    assert resubmit(context=ctx, auto=True, sleep=no_sleep) == 0
    assert len(fake_backend.calls('launch')) == before


def test_auto_submits_missing_failed_condition_probe_after_terminal_failure(
    project: Path, fake_backend: FakeBackend
) -> None:
    enable_retry(project=project, text='chained_retry = "resubmit"\n')
    study = chained_study(condition=AfterCondition.FAILED)
    write_status(
        path=project / 'runs' / 'chain' / 'pretrain' / 'a' / 'status.json',
        snapshot=RunStatusSnapshot(
            run_state=RunState.FAILED,
            updated_at=utc_now_iso(),
            instances={'train': InstanceRecord(state=InstanceState.FAILED)},
        ),
    )
    write_status(
        path=project / 'runs' / 'chain' / 'pretrain' / 'b' / 'status.json',
        snapshot=RunStatusSnapshot(
            run_state=RunState.FINISHED,
            updated_at=utc_now_iso(),
            instances={'train': InstanceRecord(state=InstanceState.DONE)},
        ),
    )
    ctx = chained_context(project=project, study=study, machine_file='cluster.toml')

    assert {report.state for report in ctx.reports if report.run.grid == 'probe'} == {
        StudyState.MISSING,
        StudyState.DEPENDENCY_FAILED,
    }
    assert resubmit(context=ctx, auto=True, sleep=no_sleep) == 0

    runs = _launched_runs(fake_backend=fake_backend)
    assert all(run.startswith('probe/') for run in runs)


def test_auto_holds_failed_condition_probe_during_upstream_retry(
    project: Path, fake_backend: FakeBackend
) -> None:
    enable_retry(project=project, text='chained_retry = "resubmit"\n')
    _time_limited(project=project, run_path='pretrain/a')
    write_status(
        path=project / 'runs' / 'chain' / 'pretrain' / 'b' / 'status.json',
        snapshot=RunStatusSnapshot(
            run_state=RunState.FINISHED,
            updated_at=utc_now_iso(),
            instances={'train': InstanceRecord(state=InstanceState.DONE)},
        ),
    )
    context = chained_context(
        project=project,
        study=chained_study(condition=AfterCondition.FAILED),
        machine_file='cluster.toml',
    )

    assert resubmit(context=context, auto=True, sleep=no_sleep) == 0

    assert _launched_runs(fake_backend=fake_backend) == ['pretrain/a']


def test_resubmit_cancels_held_job_then_resubmits(project: Path, fake_backend: FakeBackend) -> None:
    submit_study(project=project)
    fake_backend.set_states(
        phases={'job-0_0': JobPhase.QUEUED}, flags={'job-0_0': frozenset({JobFlag.REQUEUE_HELD})}
    )
    fake_backend.cancellable_answer = ['job-0']
    ctx = context(project=project, machine_file='cluster.toml')
    assert ctx.reports[0].state is StudyState.INTERRUPTED
    assert resubmit(context=ctx, sleep=no_sleep) == 0
    (cancel_call,) = fake_backend.calls('cancel')
    cancelled = cast('Context', cancel_call['context']).reports
    assert [r.run.run_path.as_posix() for r in cancelled] == ['g/small/1']
    assert 'g/small/1' in _launched_runs(fake_backend=fake_backend)


def _time_limited(*, project: Path, run_path: str) -> None:
    """A run folder left at the time limit, with a checkpoint saved."""
    folder = project / 'runs' / 'chain' / run_path
    write_status(
        path=folder / 'status.json',
        snapshot=RunStatusSnapshot(
            run_state=RunState.RUNNING,
            updated_at=utc_now_iso(),
            instances={
                'train': InstanceRecord(
                    state=InstanceState.FAILED, error='JobTimeLimitSignal: time limit'
                )
            },
        ),
    )
    (folder / 'train').mkdir(exist_ok=True)
    (folder / 'train' / 'last.ckpt').write_text('x')


def test_chain_exit_at_max_batch_requeue_ends_the_job_and_the_followup_resubmits(
    project: Path, fake_backend: FakeBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    _chain_submitted(project=project, fake_backend=fake_backend)
    enable_retry(project=project, text='chained_retry = "requeue"\n')
    records = jobs_of_chain(project=project)
    job = records[0]
    task = next(task for task in job.tasks if task.run_path.as_posix() == 'pretrain/a')
    _time_limited(project=project, run_path='pretrain/a')
    requeue = importlib.import_module('tslab.lab.study.launchers.slurm.requeue')
    chain_exit_module = importlib.import_module('tslab.lab.study.launchers.slurm.chain_exit')
    requeued: list[str] = []
    monkeypatch.setattr(requeue, 'scontrol_requeue', lambda *, task_id: requeued.append(task_id))
    monkeypatch.setattr(chain_exit_module, 'query_cluster_config', lambda: {'MaxBatchRequeue': '5'})
    # why: four restarts so far; the next requeue would reach MaxBatchRequeue
    monkeypatch.setenv('SLURM_RESTART_COUNT', '4')
    in_job = chained_context(project=project, study=chained_study(), machine_file='cluster.toml')
    assert chain_exit(context=in_job, job_file=job.job_file, index=task.index) == 1
    assert requeued == []
    assert jobs_of_chain(project=project) == records
    monkeypatch.delenv('SLURM_RESTART_COUNT')
    # the job ended non-zero; the follow-up runs resubmit --auto
    after = _chain_states(
        project=project,
        fake_backend=fake_backend,
        phases={f'{job.job_id}_{task.index}': JobPhase.FAILED},
    )
    report = next(r for r in after.reports if r.run.run_path.as_posix() == 'pretrain/a')
    assert report.state is StudyState.INTERRUPTED
    before = len(fake_backend.calls('launch'))
    assert resubmit(context=after, auto=True, sleep=no_sleep) == 0
    assert len(fake_backend.calls('launch')) == before + 1
    assert 'pretrain/a' in _launched_runs(fake_backend=fake_backend)


def test_auto_resubmit_cancels_and_resubmits_a_held_chained_job(
    project: Path, fake_backend: FakeBackend
) -> None:
    _chain_submitted(project=project, fake_backend=fake_backend)
    enable_retry(project=project, text='chained_retry = "requeue"\n')
    _time_limited(project=project, run_path='pretrain/a')
    fake_backend.cancellable_answer = ['job-0_0']
    ctx = _chain_states(
        project=project,
        fake_backend=fake_backend,
        phases={'job-0_0': JobPhase.QUEUED, 'job-0_1': JobPhase.COMPLETED},
        flags={'job-0_0': frozenset({JobFlag.REQUEUE_HELD})},
    )
    report = next(r for r in ctx.reports if r.run.run_path.as_posix() == 'pretrain/a')
    assert report.state is StudyState.INTERRUPTED
    assert resubmit(context=ctx, auto=True, sleep=no_sleep) == 0
    (cancel_call,) = fake_backend.calls('cancel')
    cancelled = cast('Context', cancel_call['context']).reports
    assert 'pretrain/a' in [r.run.run_path.as_posix() for r in cancelled]
    assert 'pretrain/a' in _launched_runs(fake_backend=fake_backend)


def jobs_of_chain(*, project: Path) -> list[JobRecord]:
    """The job records of the chained study."""
    return read_jobs(path=project / 'runs' / 'chain' / '.study' / 'submissions.jsonl')


def _cancelled_chain(
    *, project: Path, fake_backend: FakeBackend, grids: tuple[str, ...]
) -> Context:
    """The chain after ``study cancel``: pretrain/a cancelled while running, the rest queued."""
    _chain_submitted(project=project, fake_backend=fake_backend)
    # why: the cancelled run left a stale running status in its folder
    write_status(
        path=project / 'runs' / 'chain' / 'pretrain' / 'a' / 'status.json',
        snapshot=RunStatusSnapshot(
            run_state=RunState.RUNNING,
            updated_at=utc_now_iso(),
            instances={'train': InstanceRecord(state=InstanceState.RUNNING)},
        ),
    )
    task_ids = ('job-0_0', 'job-0_1', 'job-1_0', 'job-1_1', 'job-2_2', 'job-2_3')
    return _chain_states(
        project=project,
        fake_backend=fake_backend,
        phases=dict.fromkeys(task_ids, JobPhase.FAILED),
        flags=dict.fromkeys(task_ids, frozenset({JobFlag.CANCELLED})),
        grids=grids,
    )


@pytest.mark.parametrize('grids', [(), ('probe',)])
def test_default_resubmit_pulls_in_cancelled_upstream_of_dependency_failed(
    project: Path, fake_backend: FakeBackend, grids: tuple[str, ...]
) -> None:
    ctx = _cancelled_chain(project=project, fake_backend=fake_backend, grids=grids)
    states = {
        report.run.run_path.as_posix(): report.state
        for report in study_reports(context=ctx).values()
    }
    assert states['pretrain/a'] is StudyState.FAILED
    assert {state for path, state in states.items() if path.startswith('probe/')} == {
        StudyState.DEPENDENCY_FAILED
    }
    launches = len(fake_backend.calls('launch'))
    assert resubmit(context=ctx, sleep=no_sleep) == 0
    assert len(fake_backend.calls('launch')) > launches
    runs = _launched_runs(fake_backend=fake_backend)
    assert {'pretrain/a', 'pretrain/b'} <= set(runs)
    assert {run for run in runs if run.startswith('probe/')} == {
        path for path in states if path.startswith('probe/')
    }


def test_auto_resubmit_pulls_in_missing_downstream_of_selected_upstream(
    project: Path, fake_backend: FakeBackend
) -> None:
    _chain_submitted(project=project, fake_backend=fake_backend)
    enable_retry(project=project, text='chained_retry = "resubmit"\n')
    ctx = _chain_states(
        project=project,
        fake_backend=fake_backend,
        phases={'job-0_0': JobPhase.INTERRUPTED, 'job-0_1': JobPhase.COMPLETED},
        grids=('pretrain',),
    )
    folder = project / 'runs' / 'chain' / 'pretrain/a'
    (folder / 'train').mkdir(parents=True, exist_ok=True)
    (folder / 'train' / 'last.ckpt').write_text('x')
    assert resubmit(context=ctx, auto=True, sleep=no_sleep) == 0
    runs = _launched_runs(fake_backend=fake_backend)
    assert any(run.startswith('probe/a/') for run in runs)

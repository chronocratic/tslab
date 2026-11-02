"""``tslab study resubmit``: the interrupted, missing and failed runs again."""

import dataclasses
import importlib
import json
import os
from pathlib import Path
from typing import cast, TYPE_CHECKING

import pytest

from tests.support.fake_backend import FakeBackend
from tests.unit.study.helpers import (
    context,
    finish_run,
    jobs_of,
    no_sleep,
    record_local_job,
    study_directory,
    submit_study,
    write_run_status,
)
from tslab.core.meta import hostname
from tslab.core.paths import LOCK_FILE
from tslab.core.status import (
    InstanceRecord,
    InstanceState,
    RunState,
    RunStatusSnapshot,
    utc_now_iso,
    write_status,
)
from tslab.study.code_snapshot import CodeSnapshot, CodeSnapshotError
from tslab.study.commands import resubmit, submit, submit_runs
from tslab.study.commands.resubmit import _remove_ended_locks
from tslab.study.context import Context, load_context, refresh_context, Selection
from tslab.study.launchers import JobPhase, LaunchError
from tslab.study.reporting.state import RunReport, StudyState

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tslab.study.plan import JobGroup


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
    assert [t.run_path for t in jobs[1].tasks] == ['g/large/1', 'g/large/2']
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
    assert [t.run_path for t in jobs[1].tasks] == ['g/small/2']


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
    assert all(t.run_path != 'g/small/1' for t in jobs[1].tasks)
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
    assert [t.run_path for t in jobs[1].tasks] == ['g/small/1']
    assert fake_backend.calls('launch')[-1]['start_delay_seconds'] == 660


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
    assert 'g/large/2' in [t.run_path for t in jobs[1].tasks]


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
    (project / 'runs' / 's' / 'g/small/1' / 'last.ckpt').write_text('ckpt')
    fake_backend.set_states(phases={'job-0_0': JobPhase.COMPLETED})
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
        local.read_text() + f'\n[followup]\nauto_retry = true\nmax_attempts = {max_attempts}\n'
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


def test_local_auto_retry_finishes_after_three_attempts(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('TOY_SUCCEED_ON', '3')
    ctx = _local_retry_context(project=project, max_attempts=3)
    submit(context=ctx, include_done=False, dry_run=False, extra_args=(), sleep=no_sleep)
    assert _local_attempts(project=project) == 3
    report = refresh_context(context=ctx).reports[0]
    assert report.state is StudyState.DONE
    assert len(jobs_of(project=project)) == 3


def test_local_auto_retry_stops_at_max_attempts(
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
    submit_module = importlib.import_module('tslab.study.commands.submit')
    monkeypatch.setattr(submit_module, 'submit_runs', lambda **_: (143, []))
    monkeypatch.setattr('tslab.study.commands.resubmit', lambda **kwargs: calls.append(kwargs))
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
            'g/small/1': first,
            'g/small/2': second,
            'g/large/1': first,
            'g/large/2': None,
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
        'g/small/1': first.folder,
        'g/small/2': second.folder,
        'g/large/1': first.folder,
        'g/large/2': None,
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
            'g/small/1': first,
            'g/small/2': first,
            'g/large/1': second,
            'g/large/2': second,
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
    (project / 'runs' / 's' / 'g/small/1' / 'last.ckpt').write_text('ckpt')
    fake_backend.set_states(phases={'job-0_0': JobPhase.COMPLETED})
    code = resubmit(
        context=context(project=project, machine_file='cluster.toml'), auto=True, sleep=no_sleep
    )
    assert code == 0
    latest = jobs_of(project=project)[-1]
    assert [task.run_path for task in latest.tasks] == ['g/small/1']
    assert latest.code_snapshot is not None
    assert latest.code_snapshot.folder == snapshot.folder

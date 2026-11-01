"""``tslab study resubmit``: the interrupted, missing and failed runs again."""

import dataclasses
import json
from pathlib import Path

import pytest

from tests.support.fake_slurm import FakeSlurm
from tests.unit.study.helpers import (
    context,
    finish_run,
    jobs_of,
    no_sleep,
    submit_study,
    write_run_status,
)
from tslab.core.status import (
    InstanceRecord,
    InstanceState,
    RunState,
    RunStatusSnapshot,
    utc_now_iso,
    write_status,
)
from tslab.study.commands import resubmit, submit
from tslab.study.context import Context, load_context, refresh_context, Selection
from tslab.study.state import StudyState


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


def test_default_states_are_interrupted_and_missing(project: Path, fake: FakeSlurm) -> None:
    submit_study(project=project)
    fake.set_answers('squeue', [{'exit': 0, 'stdout': ''}])
    fake.set_answers(
        'sacct',
        [
            {
                'exit': 0,
                'stdout': ('1000_0|COMPLETED\n1000_1|FAILED\n1000_2|TIMEOUT\n1000_3|COMPLETED\n'),
            }
        ],
    )
    finish_run(project=project, model='small', seed=1)
    code = resubmit(context=context(project=project, machine_file='cluster.toml'), sleep=no_sleep)
    assert code == 0
    jobs = jobs_of(project=project)
    assert [t.run_path for t in jobs[1].tasks] == ['g/large/1', 'g/large/2']
    assert all(t.attempt == 2 for t in jobs[1].tasks)


def test_failed_must_be_asked_for(project: Path, fake: FakeSlurm) -> None:
    submit_study(project=project)
    fake.set_answers('sacct', [{'exit': 0, 'stdout': '1000_1|FAILED\n'}])
    code = resubmit(
        context=context(project=project, machine_file='cluster.toml'),
        states=(StudyState.FAILED,),
        sleep=no_sleep,
    )
    assert code == 0
    jobs = jobs_of(project=project)
    assert [t.run_path for t in jobs[1].tasks] == ['g/small/2']


def test_resubmit_never_touches_live_runs(
    project: Path, fake: FakeSlurm, caplog: pytest.LogCaptureFixture
) -> None:
    submit_study(project=project)
    fake.set_answers('squeue', [{'exit': 0, 'stdout': '1000_0|RUNNING\n'}])
    all_states = tuple(StudyState)
    code = resubmit(
        context=context(project=project, machine_file='cluster.toml'),
        states=all_states,
        sleep=no_sleep,
    )
    assert code == 0
    jobs = jobs_of(project=project)
    assert all(t.run_path != 'g/small/1' for t in jobs[1].tasks)
    assert fake.calls('scancel') == []


def test_cancel_live_cancels_then_resubmits_later(project: Path, fake: FakeSlurm) -> None:
    submit_study(project=project)
    fake.set_answers('squeue', [{'exit': 0, 'stdout': '1000_0|PENDING\n'}])
    fake.set_answers(
        'sacct', [{'exit': 0, 'stdout': '1000_1|COMPLETED\n1000_2|COMPLETED\n1000_3|COMPLETED\n'}]
    )
    for model, seed in (('small', 2), ('large', 1), ('large', 2)):
        finish_run(project=project, model=model, seed=seed)
    code = resubmit(
        context=context(project=project, machine_file='cluster.toml'),
        cancel_live=True,
        sleep=no_sleep,
    )
    assert code == 0
    assert fake.calls('scancel') == [['1000_0']]
    jobs = jobs_of(project=project)
    assert [t.run_path for t in jobs[1].tasks] == ['g/small/1']
    script = (project / 'runs' / 's' / '.study' / 'jobs' / f'{jobs[1].job_file}.sbatch').read_text()
    assert '#SBATCH --begin=now+660' in script


@pytest.mark.slow
def test_cancel_live_never_cancels_local_jobs(project: Path, fake: FakeSlurm) -> None:
    # why: local job ids (local-<pid>-<n>) are not SLURM jobs
    code = submit(
        context=context(project=project, machine_file='local.toml'),
        include_done=False,
        dry_run=False,
        extra_args=(),
        sleep=no_sleep,
    )
    assert code == 0
    (project / 'runs' / 's' / 'g/small/1' / '.lock').write_text('')
    code = resubmit(
        context=context(project=project, machine_file='local.toml'),
        cancel_live=True,
        sleep=no_sleep,
    )
    assert code == 0
    assert fake.calls('scancel') == []


def test_ended_job_lock_is_removed_before_resubmitting(
    project: Path, fake: FakeSlurm, caplog: pytest.LogCaptureFixture
) -> None:
    submit_study(project=project)
    fake.set_answers('sacct', [{'exit': 0, 'stdout': '1000_3|TIMEOUT\n'}])
    running_run_with_lock(project=project, model='large', seed=2)
    with caplog.at_level('INFO'):
        code = resubmit(
            context=context(project=project, machine_file='cluster.toml'), sleep=no_sleep
        )
    assert code == 0
    lock = project / 'runs' / 's' / 'g/large/2' / '.lock'
    assert not lock.exists()
    assert 'removed the lock of g/large/2: its job 1000_3 ended (TIMEOUT)' in caplog.text
    jobs = jobs_of(project=project)
    assert 'g/large/2' in [t.run_path for t in jobs[1].tasks]


def test_pass_through_arguments_reach_jobs_and_retries(
    project: Path, fake: FakeSlurm, tmp_path: Path
) -> None:
    track = tmp_path / 'track.log'
    submit_study(project=project, extra_args=('--tracking-log', str(track)))
    first = jobs_of(project=project)[0]
    assert first.tasks[0].extra_args == ('--tracking-log', str(track))
    assert first.tasks[0].command[-2:] == ('--tracking-log', str(track))
    fake.set_answers('sacct', [{'exit': 0, 'stdout': ''}])
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


def live_cancel_setup(*, project: Path, fake: FakeSlurm) -> None:
    """Task 0 queued and tasks 1 to 3 done, so only one run is live."""
    submit_study(project=project)
    fake.set_answers('squeue', [{'exit': 0, 'stdout': '1000_0|PENDING\n'}])
    fake.set_answers(
        'sacct', [{'exit': 0, 'stdout': '1000_1|COMPLETED\n1000_2|COMPLETED\n1000_3|COMPLETED\n'}]
    )
    for model, seed in (('small', 2), ('large', 1), ('large', 2)):
        finish_run(project=project, model=model, seed=seed)


def test_dry_run_with_cancel_live_cancels_and_unlocks_nothing(
    project: Path, fake: FakeSlurm, caplog: pytest.LogCaptureFixture
) -> None:
    live_cancel_setup(project=project, fake=fake)
    fake.set_answers('sacct', [{'exit': 0, 'stdout': '1000_3|TIMEOUT\n'}])
    running_run_with_lock(project=project, model='large', seed=2)
    with caplog.at_level('INFO'):
        code = resubmit(
            context=context(project=project, machine_file='cluster.toml'),
            cancel_live=True,
            dry_run=True,
            sleep=no_sleep,
        )
    assert code == 0
    assert fake.calls('scancel') == []
    assert (project / 'runs' / 's' / 'g/large/2' / '.lock').exists()
    assert 'would cancel 1000_0' in caplog.text
    assert len(jobs_of(project=project)) == 1


def test_failed_validation_cancels_nothing(project: Path, fake: FakeSlurm) -> None:
    live_cancel_setup(project=project, fake=fake)
    # why: a pass-through flag the config does not have makes every run fail validation
    code = resubmit(
        context=context(project=project, machine_file='cluster.toml'),
        cancel_live=True,
        extra_args=('--no-such-flag',),
        sleep=no_sleep,
    )
    assert code == 1
    assert fake.calls('scancel') == []
    assert len(jobs_of(project=project)) == 1


def test_cancel_failure_exits_one_and_submits_nothing(project: Path, fake: FakeSlurm) -> None:
    live_cancel_setup(project=project, fake=fake)
    fake.set_answers('scancel', [{'exit': 1, 'stderr': 'Invalid job id'}])
    code = resubmit(
        context=context(project=project, machine_file='cluster.toml'),
        cancel_live=True,
        sleep=no_sleep,
    )
    assert code == 1
    assert len(fake.calls('sbatch')) == 1
    assert len(jobs_of(project=project)) == 1


def test_cancel_live_passes_only_live_job_ids(project: Path, fake: FakeSlurm) -> None:
    live_cancel_setup(project=project, fake=fake)
    fake.set_answers('sacct', [{'exit': 0, 'stdout': '1000_1|TIMEOUT\n1000_2|COMPLETED\n'}])
    code = resubmit(
        context=context(project=project, machine_file='cluster.toml'),
        cancel_live=True,
        sleep=no_sleep,
    )
    assert code == 0
    # why: the timed out job already ended; scancel on an ended id would fail
    assert fake.calls('scancel') == [['1000_0']]


def test_unknown_slurm_states_refuse_to_submit(
    project: Path, fake: FakeSlurm, caplog: pytest.LogCaptureFixture
) -> None:
    submit_study(project=project)
    ctx = context(project=project, machine_file='cluster.toml')
    unknown = dataclasses.replace(ctx, job_states_known=False)
    with caplog.at_level('ERROR'):
        code = resubmit(context=unknown, states=tuple(StudyState), sleep=no_sleep)
    assert code == 1
    assert len(fake.calls('sbatch')) == 1
    assert 'not submitting' in caplog.text


def test_auto_resubmit_retries_paused_run_with_progress(project: Path, fake: FakeSlurm) -> None:
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
    fake.set_answers('squeue', [{'exit': 0, 'stdout': ''}])
    fake.set_answers('sacct', [{'exit': 0, 'stdout': '1000_0|COMPLETED\n'}])
    code = resubmit(
        context=context(project=project, machine_file='cluster.toml'),
        auto=True,
        dry_run=True,
        sleep=no_sleep,
    )
    assert code == 0
    # the dry-run plan holds only the paused run, one attempt further on
    tasks_files = sorted((project / 'runs' / 's' / '.study' / 'jobs').glob('*.tasks'))
    planned = tasks_files[-1].read_text()
    assert planned.startswith('0 2 ')
    assert '--run-path g/small/1' in planned
    assert 'g/small/2' not in planned
    assert 'g/large' not in planned
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
    monkeypatch.setattr('tslab.study.commands.submit_runs', lambda **_: (143, []))
    monkeypatch.setattr('tslab.study.commands.resubmit', lambda **kwargs: calls.append(kwargs))
    code = submit(context=ctx, include_done=False, dry_run=False, extra_args=(), sleep=no_sleep)
    assert code == 143
    assert calls == []

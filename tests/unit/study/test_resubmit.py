"""``tslab study resubmit``: the interrupted, missing and failed runs again."""

import json
from pathlib import Path

import pytest

from study.fake_slurm import FakeSlurm, install_fake_slurm
from study.project import write_toy_project
from tslab.core.run import _LOCK_FILE  # why: the lock name is private to tslab.core.run
from tslab.core.status import (
    InstanceRecord,
    InstanceState,
    RunState,
    RunStatusSnapshot,
    utc_now_iso,
    write_status,
)
from tslab.study import (
    Context,
    JobRecord,
    load_context,
    read_jobs,
    resubmit,
    Selection,
    StudyState,
    submit,
)


def no_sleep(_seconds: float) -> None:
    """Skip the submission pause."""


def context(*, project: Path, machine_file: str) -> Context:
    return load_context(
        reference='toy_studies:study',
        machine_path=project / machine_file,
        selection=Selection(),
        project_root=Path.cwd(),
    )


def submit_study(*, project: Path, extra_args: tuple[str, ...] = ()) -> None:
    code = submit(
        context=context(project=project, machine_file='cluster.toml'),
        include_done=False,
        dry_run=False,
        extra_args=extra_args,
        sleep=no_sleep,
    )
    assert code == 0


def finish_run(*, project: Path, model: str, seed: int) -> None:
    """Write a done status.json, so the run reads as done from its folder."""
    folder = project / 'runs' / 's' / f'g/{model}/{seed}'
    folder.mkdir(parents=True)
    write_status(
        path=folder / 'status.json',
        snapshot=RunStatusSnapshot(
            run_state=RunState.FINISHED,
            updated_at=utc_now_iso(),
            instances={'train': InstanceRecord(state=InstanceState.DONE)},
        ),
    )


def running_run_with_lock(*, project: Path, model: str, seed: int) -> None:
    """A run folder that says running and still holds a live foreign lock."""
    folder = project / 'runs' / 's' / f'g/{model}/{seed}'
    folder.mkdir(parents=True)
    write_status(
        path=folder / 'status.json',
        snapshot=RunStatusSnapshot(
            run_state=RunState.RUNNING,
            updated_at=utc_now_iso(),
            instances={'train': InstanceRecord(state=InstanceState.RUNNING)},
        ),
    )
    (folder / '.lock').write_text(json.dumps({'pid': 99999, 'host': 'otherhost'}))


def jobs_of(*, project: Path) -> list[JobRecord]:
    return read_jobs(path=project / 'runs' / 's' / '.study' / 'submissions.jsonl')


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    folder = write_toy_project(project=tmp_path / 'proj')
    # why: load_study imports from the current directory, like a user in a project
    monkeypatch.chdir(folder)
    return folder


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeSlurm:
    return install_fake_slurm(tmp_path=tmp_path, monkeypatch=monkeypatch, answers={})


def test_lock_file_name_is_pinned() -> None:
    # why: resubmit removes stale locks by this literal name
    assert _LOCK_FILE == '.lock'


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

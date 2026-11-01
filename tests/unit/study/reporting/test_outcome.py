"""Exit codes for study states and pending work."""

from pathlib import Path
import subprocess
import sys
from typing import Literal

import pytest

from tests.support.fake_slurm import FakeSlurm
from tests.unit.study.helpers import (
    context,
    finish_run,
    jobs_of,
    study_directory,
    submit_study,
    write_run_status,
)
from tslab.core.status import InstanceState, RunState, utc_now_iso
from tslab.study.launchers.slurm.states import followup_job_ids
from tslab.study.records import append_job, JobRecord, submission_in_progress
from tslab.study.reporting.outcome import (
    EXIT_DONE,
    EXIT_LIVE,
    EXIT_RETRY,
    EXIT_UNKNOWN,
    states_exit_code,
    study_exit_code,
)
from tslab.study.reporting.state import StudyState


def _job_record(
    *, kind: Literal['run', 'followup'], launcher: Literal['local', 'slurm'], job_id: str
) -> JobRecord:
    return JobRecord(
        kind=kind,
        submission_id=job_id,
        time=utc_now_iso(),
        launcher=launcher,
        job_id=job_id,
        job_file=1,
        tasks=(),
    )


def test_all_done_exits_0(project: Path, fake: FakeSlurm) -> None:
    submit_study(project=project)
    for model in ('small', 'large'):
        for seed in (1, 2):
            finish_run(project=project, model=model, seed=seed)
    assert (
        study_exit_code(context=context(project=project, machine_file='cluster.toml')) == EXIT_DONE
    )


def test_empty_states_exit_0() -> None:
    assert states_exit_code(states=[], live=False) == EXIT_DONE


def test_failed_beats_running() -> None:
    assert states_exit_code(states=[StudyState.FAILED, StudyState.RUNNING], live=False) == 4


def test_running_beats_missing() -> None:
    assert states_exit_code(states=[StudyState.RUNNING, StudyState.MISSING], live=False) == 3


def test_missing_only_needs_retry() -> None:
    assert states_exit_code(states=[StudyState.DONE, StudyState.MISSING], live=False) == EXIT_RETRY


def test_unknown_slurm_states_exit_6(
    project: Path, fake: FakeSlurm, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr('tslab.study.launchers.slurm.backend.slurm_available', lambda: False)
    assert (
        study_exit_code(context=context(project=project, machine_file='cluster.toml'))
        == EXIT_UNKNOWN
    )


def test_recorded_run_without_folder_is_live_while_lock_held(
    project: Path, fake: FakeSlurm
) -> None:
    submit_study(project=project)
    lock_file = study_directory(project=project) / '.study' / 'submit.lock'
    code = (
        'import fcntl, sys, time\n'
        'with open(sys.argv[1], "r+") as lock:\n'
        '    fcntl.flock(lock, fcntl.LOCK_EX)\n'
        '    print("ready", flush=True)\n'
        '    time.sleep(30)\n'
    )
    process = subprocess.Popen(  # noqa: S603 # why: fixed test code locks tmp_path file
        [sys.executable, '-c', code, str(lock_file)], stdout=subprocess.PIPE, text=True
    )
    assert process.stdout is not None
    assert process.stdout.readline() == 'ready\n'
    try:
        assert (
            study_exit_code(context=context(project=project, machine_file='cluster.toml'))
            == EXIT_LIVE
        )
    finally:
        process.terminate()
        process.wait(timeout=5)
        process.stdout.close()
    assert (
        study_exit_code(context=context(project=project, machine_file='cluster.toml')) == EXIT_RETRY
    )


def test_interrupted_run_is_live_while_followup_queued(project: Path, fake: FakeSlurm) -> None:
    submit_study(project=project)
    job = jobs_of(project=project)[0]
    task = job.tasks[0]
    _, model, seed = task.run_path.split('/')
    write_run_status(
        project=project,
        model=model,
        seed=int(seed),
        run_state=RunState.RUNNING,
        instance=InstanceState.RUNNING,
    )
    followup = _job_record(kind='followup', launcher='slurm', job_id='2000')
    append_job(
        path=study_directory(project=project) / '.study' / 'submissions.jsonl', record=followup
    )
    fake.set_answers(
        'sacct', [{'exit': 0, 'stdout': f'{job.job_id}_{task.index}|TIMEOUT\n2000|PENDING\n'}]
    )
    assert (
        study_exit_code(context=context(project=project, machine_file='cluster.toml')) == EXIT_LIVE
    )


def test_interrupted_runs_are_live_while_resubmit_holds_lock(
    project: Path, fake: FakeSlurm
) -> None:
    submit_study(project=project)
    job = jobs_of(project=project)[0]
    for task in job.tasks:
        _, model, seed = task.run_path.split('/')
        write_run_status(
            project=project,
            model=model,
            seed=int(seed),
            run_state=RunState.RUNNING,
            instance=InstanceState.RUNNING,
        )
    fake.set_answers(
        'sacct',
        [
            {
                'exit': 0,
                'stdout': ''.join(f'{job.job_id}_{task.index}|TIMEOUT\n' for task in job.tasks),
            }
        ],
    )
    lock_file = study_directory(project=project) / '.study' / 'submit.lock'
    code = (
        'import fcntl, sys, time\n'
        'with open(sys.argv[1], "r+") as lock:\n'
        '    fcntl.flock(lock, fcntl.LOCK_EX)\n'
        '    print("ready", flush=True)\n'
        '    time.sleep(30)\n'
    )
    process = subprocess.Popen(  # noqa: S603 # why: fixed test code locks tmp_path file
        [sys.executable, '-c', code, str(lock_file)], stdout=subprocess.PIPE, text=True
    )
    assert process.stdout is not None
    assert process.stdout.readline() == 'ready\n'
    try:
        assert (
            study_exit_code(context=context(project=project, machine_file='cluster.toml'))
            == EXIT_LIVE
        )
    finally:
        process.terminate()
        process.wait(timeout=5)
        process.stdout.close()


def test_submission_in_progress_false_without_lock_file(tmp_path: Path) -> None:
    assert submission_in_progress(study_folder=tmp_path) is False
    assert not (tmp_path / 'submit.lock').exists()


def test_followup_job_ids_lists_only_sorted_slurm_followups() -> None:
    jobs = [
        _job_record(kind='followup', launcher='slurm', job_id='20'),
        _job_record(kind='run', launcher='slurm', job_id='10'),
        _job_record(kind='followup', launcher='local', job_id='local-123-1'),
        _job_record(kind='followup', launcher='slurm', job_id='3'),
    ]
    assert followup_job_ids(jobs=jobs) == ['20', '3']

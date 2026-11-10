"""Exit codes for study states and pending work."""

from pathlib import Path
import subprocess
import sys

from tests.lab.unit.study.helpers import (
    context,
    finish_run,
    jobs_of,
    study_directory,
    submit_study,
    write_run_status,
)
from tests.support.fake_backend import FakeBackend
from tslab.lab.core.status import utc_now_iso
from tslab.lab.enums import InstanceState, JobPhase, LauncherKind, RunState, StudyState
from tslab.lab.study.launchers.tables import backend_for, JOB_RECORDS
from tslab.lab.study.records import append_job, JobKind, JobRecord, submission_in_progress
from tslab.lab.study.reporting.outcome import (
    EXIT_DONE,
    EXIT_LIVE,
    EXIT_RETRY,
    EXIT_UNKNOWN,
    states_exit_code,
    study_exit_code,
)


def _job_record(*, kind: JobKind, launcher: LauncherKind, job_id: str) -> JobRecord:
    return JOB_RECORDS[launcher](
        kind=kind,
        submission_id=job_id,
        time=utc_now_iso(),
        launcher=launcher,
        job_id=job_id,
        job_file=1,
        tasks=(),
    )


def test_fake_backend_serves_both_launchers(project: Path, fake_backend: FakeBackend) -> None:
    for machine_file in ('cluster.toml', 'local.toml'):
        machine = context(project=project, machine_file=machine_file).machine
        assert backend_for(machine=machine) is fake_backend


def test_all_done_exits_0(project: Path, fake_backend: FakeBackend) -> None:
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


def test_unknown_slurm_states_exit_6(project: Path, fake_backend: FakeBackend) -> None:
    fake_backend.set_states(phases={}, known=False)
    assert (
        study_exit_code(context=context(project=project, machine_file='cluster.toml'))
        == EXIT_UNKNOWN
    )


def test_recorded_run_without_folder_is_live_while_lock_held(
    project: Path, fake_backend: FakeBackend
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


def test_interrupted_run_is_live_while_followup_queued(
    project: Path, fake_backend: FakeBackend
) -> None:
    submit_study(project=project)
    job = jobs_of(project=project)[0]
    task = job.tasks[0]
    _, model, seed = task.run_path.parts
    write_run_status(
        project=project,
        model=model,
        seed=int(seed),
        run_state=RunState.RUNNING,
        instance=InstanceState.RUNNING,
    )
    followup = _job_record(kind=JobKind.FOLLOWUP, launcher=LauncherKind.SLURM, job_id='job-9')
    append_job(
        path=study_directory(project=project) / '.study' / 'submissions.jsonl', record=followup
    )
    fake_backend.set_states(
        phases={f'{job.job_id}_{task.index}': JobPhase.INTERRUPTED},
        followup_live_job_ids=('job-9',),
    )
    assert (
        study_exit_code(context=context(project=project, machine_file='cluster.toml')) == EXIT_LIVE
    )


def test_interrupted_runs_are_live_while_resubmit_holds_lock(
    project: Path, fake_backend: FakeBackend
) -> None:
    submit_study(project=project)
    job = jobs_of(project=project)[0]
    for task in job.tasks:
        _, model, seed = task.run_path.parts
        write_run_status(
            project=project,
            model=model,
            seed=int(seed),
            run_state=RunState.RUNNING,
            instance=InstanceState.RUNNING,
        )
    fake_backend.set_states(
        phases={f'{job.job_id}_{task.index}': JobPhase.INTERRUPTED for task in job.tasks}
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

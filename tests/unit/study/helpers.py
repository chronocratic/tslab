"""Helpers shared by the study command tests: contexts, submissions, run folders."""

from pathlib import Path
import subprocess
import sys

from tslab.core.files import atomic_write_json
from tslab.core.status import (
    InstanceRecord,
    InstanceState,
    RunState,
    RunStatusSnapshot,
    utc_now_iso,
    write_status,
)
from tslab.study.commands import submit
from tslab.study.context import Context, load_context, Selection
from tslab.study.local_launch import LAUNCH_FILE
from tslab.study.records import append_job, JobRecord, ProgressSnapshot, read_jobs, TaskRecord

__all__ = [
    'context',
    'finish_run',
    'finished_pid',
    'jobs_of',
    'no_sleep',
    'record_local_job',
    'study_directory',
    'submit_study',
    'write_launch',
    'write_run_status',
]


def no_sleep(_seconds: float) -> None:
    """Skip the submission pause."""


def study_directory(*, project: Path) -> Path:
    """The folder of the toy study ``s`` under the project."""
    return project / 'runs' / 's'


def context(*, project: Path, machine_file: str) -> Context:
    """The context of ``toy_studies:study`` on one machine profile of the project."""
    return load_context(
        reference='toy_studies:study',
        machine_path=project / machine_file,
        selection=Selection(),
        project_root=Path.cwd(),
    )


def submit_study(
    *, project: Path, machine_file: str = 'cluster.toml', extra_args: tuple[str, ...] = ()
) -> None:
    """Submit the whole toy study and check that it went through."""
    code = submit(
        context=context(project=project, machine_file=machine_file),
        include_done=False,
        dry_run=False,
        extra_args=extra_args,
        sleep=no_sleep,
    )
    assert code == 0


def jobs_of(*, project: Path) -> list[JobRecord]:
    """The job records of the toy study."""
    return read_jobs(path=study_directory(project=project) / '.study' / 'submissions.jsonl')


def write_run_status(  # noqa: PLR0913 # why: one flag per status field, all keyword-only
    *,
    project: Path,
    model: str,
    seed: int,
    run_state: RunState,
    instance: InstanceState,
    error: str | None = None,
) -> Path:
    """Write a ``status.json`` for one toy run and return its folder."""
    folder = study_directory(project=project) / f'g/{model}/{seed}'
    folder.mkdir(parents=True)
    write_status(
        path=folder / 'status.json',
        snapshot=RunStatusSnapshot(
            run_state=run_state,
            updated_at=utc_now_iso(),
            instances={'train': InstanceRecord(state=instance, error=error)},
        ),
    )
    return folder


def finish_run(*, project: Path, model: str, seed: int) -> None:
    """Write a done status.json, so the run reads as done from its folder."""
    write_run_status(
        project=project,
        model=model,
        seed=seed,
        run_state=RunState.FINISHED,
        instance=InstanceState.DONE,
    )


def write_launch(*, folder: Path, pid: int, host: str, tasks: dict[str, str]) -> None:
    """Write a launch file as ``hold_launch`` would."""
    folder.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        path=folder / LAUNCH_FILE,
        document={
            'pid': pid,
            'hostname': host,
            'started': '2026-01-01T00:00:00+00:00',
            'tasks': [[key, state] for key, state in tasks.items()],
        },
    )


def record_local_job(*, project: Path) -> list[str]:
    """Record one local job holding every run of the toy study; return the task ids."""
    ctx = context(project=project, machine_file='local.toml')
    tasks = tuple(
        TaskRecord(
            index=index,
            grid=run.grid,
            run_path=run.run_path,
            command=('true',),
            extra_args=(),
            resources={},
            attempt=1,
            progress=ProgressSnapshot(done_instances=0, newest_checkpoint_mtime=None),
        )
        for index, run in enumerate(ctx.runs)
    )
    append_job(
        path=study_directory(project=project) / '.study' / 'submissions.jsonl',
        record=JobRecord(
            kind='run',
            submission_id='s1',
            time=utc_now_iso(),
            launcher='local',
            job_id='local-1-1',
            job_file=1,
            tasks=tasks,
        ),
    )
    return [f'local-1-1_{task.index}' for task in tasks]


def finished_pid() -> int:
    """The pid of a child process that has already exited."""
    child = subprocess.Popen([sys.executable, '-c', 'pass'])  # why: fixed command
    child.wait()
    return child.pid

"""Helpers shared by the study command tests: contexts, submissions, run folders."""

from pathlib import Path

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
from tslab.study.records import JobRecord, read_jobs

__all__ = [
    'context',
    'finish_run',
    'jobs_of',
    'no_sleep',
    'study_directory',
    'submit_study',
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


def write_run_status(
    *, project: Path, model: str, seed: int, run_state: RunState, instance: InstanceState
) -> Path:
    """Write a ``status.json`` for one toy run and return its folder."""
    folder = study_directory(project=project) / f'g/{model}/{seed}'
    folder.mkdir(parents=True)
    write_status(
        path=folder / 'status.json',
        snapshot=RunStatusSnapshot(
            run_state=run_state,
            updated_at=utc_now_iso(),
            instances={'train': InstanceRecord(state=instance)},
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

"""Submitting planned job groups: the job files, and the launch itself."""

from collections.abc import Callable, Mapping, Sequence
import dataclasses
from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import socket
import subprocess
import time
from typing import Literal, Protocol
from uuid import uuid4

from tslab.core.paths import atomic_write_text
from tslab.core.status import utc_now_iso
from tslab.study.definition import Study
from tslab.study.machine import MachineProfile
from tslab.study.plan import JobGroup, study_folder
from tslab.study.records import append_job, JobRecord, take_progress, TaskRecord
from tslab.study.slurm import job_script, sbatch, SlurmError

__all__ = [
    'LaunchReport',
    'Launcher',
    'LocalLauncher',
    'SlurmLauncher',
    'launch',
    'write_job_files',
]


@dataclass(frozen=True, kw_only=True)
class LaunchReport:
    """What one launch submitted.

    Args:
        submitted: One job record per submitted group, in order.
        not_submitted: The groups the launch did not reach; all of them
            on a dry run, the rest after an error.
        error: The submission error that stopped the launch, if any.
    """

    submitted: tuple[JobRecord, ...]
    not_submitted: tuple[JobGroup, ...]
    error: str | None


def write_job_files(  # noqa: PLR0913 # why: keyword-only public API; each option is documented below
    *,
    study_folder: Path,
    groups: Sequence[JobGroup],
    machine: MachineProfile,
    project_root: Path,
    dependency: str | None = None,
    begin: str | None = None,
) -> None:
    """The ``jobs/<n>.tasks`` and ``jobs/<n>.sbatch`` files of every group.

    Args:
        study_folder: The study's ``.study`` folder.
        groups: The planned groups.
        machine: The machine the jobs run on; its ``[env]`` goes in the scripts.
        project_root: The directory the scripts ``cd`` into.
        dependency: A SLURM ``--dependency``, for follow-up jobs.
        begin: A SLURM ``--begin``, for follow-up jobs.
    """
    (study_folder / 'jobs').mkdir(parents=True, exist_ok=True)
    (study_folder / 'logs').mkdir(parents=True, exist_ok=True)
    for group in groups:
        tasks_text = '\n'.join(
            f'{task.index} {task.attempt} {shlex.join(task.command)}' for task in group.tasks
        )
        atomic_write_text(
            path=study_folder / 'jobs' / f'{group.job_file}.tasks', text=tasks_text + '\n'
        )
        if group.slurm is None:
            continue
        script = job_script(
            job_name=group.job_name,
            slurm=group.slurm,
            indices=[task.index for task in group.tasks],
            tasks_file=study_folder / 'jobs' / f'{group.job_file}.tasks',
            log_pattern=study_folder / 'logs' / f'{group.job_name}_%a.out',
            project_root=project_root,
            env=machine.env,
            dependency=dependency,
            begin=begin,
        )
        atomic_write_text(path=study_folder / 'jobs' / f'{group.job_file}.sbatch', text=script)


class Launcher(Protocol):
    """What one launch does with a written group: returns the job id."""

    def submit(self, *, group: JobGroup, study_folder: Path) -> str:
        """Submit one group's job files and return the job id."""


class SlurmLauncher:
    """One ``sbatch`` per group, paused between the cluster's limits."""

    def __init__(self, *, sleep: Callable[[float], None] = time.sleep) -> None:
        """Args: sleep: What waits out the cluster's submission pause."""
        self._sleep = sleep

    def submit(self, *, group: JobGroup, study_folder: Path) -> str:
        """The id of the group's ``sbatch`` job."""
        job_id = sbatch(script=study_folder / 'jobs' / f'{group.job_file}.sbatch')
        if group.slurm is not None:
            self._sleep(group.slurm.submit_pause_seconds)
        return job_id


class LocalLauncher:
    """The group's tasks, one after another, in the foreground."""

    def __init__(self, *, project_root: Path, env: Mapping[str, str]) -> None:
        """Args: project_root: The directory the tasks run in; env: the machine's ``[env]``."""
        self._project_root = project_root
        self._env = env

    def submit(self, *, group: JobGroup, study_folder: Path) -> str:  # noqa: ARG002 # why: the Protocol signature
        """Run every task, continuing after a failing one; the job id.

        Each task appends to its log file — header, then the output line by
        line, which the console also gets — and a non-zero exit is logged
        without stopping the group.
        """
        job_id = f'local-{os.getpid()}-{group.job_file}'
        for task in group.tasks:
            task.log_file.parent.mkdir(parents=True, exist_ok=True)
            with task.log_file.open('a', encoding='utf-8') as log:
                header = (
                    f'=== tslab attempt {task.attempt} | job {job_id} '
                    f'| {socket.gethostname()} | {utc_now_iso()} ===\n'
                )
                log.write(header)
                log.flush()
                process = subprocess.Popen(  # noqa: S603 # why: the run command, typed by the study
                    list(task.command),  # why: its first word is the user's script command
                    cwd=self._project_root,
                    env={**os.environ, **self._env},
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                stdout = process.stdout
                if stdout is None:  # pragma: no cover — PIPE always has a stream
                    message = f'no output stream for task {task.index}'
                    raise RuntimeError(message)
                for line in stdout:
                    print(line, end='')  # noqa: T201 # why: the console output is the design
                    log.write(line)
                log.flush()
                exit_code = process.wait()
            if exit_code != 0:
                line = f'=== task {task.index} exited with {exit_code} ===\n'
                print(line, end='')  # noqa: T201 # why: the console output is the design
                with task.log_file.open('a', encoding='utf-8') as log:
                    log.write(line)
        return job_id


def launch(  # noqa: PLR0913 # why: keyword-only public API; each option is documented below
    *,
    launcher: Launcher,
    machine: MachineProfile,
    study: Study,
    groups: Sequence[JobGroup],
    project_root: Path,
    dry_run: bool,
    kind: Literal['run'] = 'run',
    dependency: str | None = None,
    begin: str | None = None,
) -> LaunchReport:
    """Write the job files, then submit the groups one by one.

    A dry run writes and stops. Otherwise each submitted group is recorded
    in ``submissions.jsonl`` before the next is submitted, so a failing
    submission leaves exactly the records of the jobs that were sent; the
    launch stops there and reports the rest as not submitted.

    Args:
        launcher: What submits the written groups.
        machine: The machine the jobs run on.
        study: The study the groups came from.
        groups: The planned groups, in submission order.
        project_root: The directory the jobs run in.
        dry_run: Write the job files and stop, submitting nothing.
        kind: The kind recorded on each job; ``run`` for study runs.
        dependency: A SLURM ``--dependency``, for follow-up jobs.
        begin: A SLURM ``--begin``, for follow-up jobs.
    """
    folder = study_folder(machine=machine, study=study)
    write_job_files(
        study_folder=folder,
        groups=groups,
        machine=machine,
        project_root=project_root,
        dependency=dependency,
        begin=begin,
    )
    if dry_run:
        return LaunchReport(submitted=(), not_submitted=tuple(groups), error=None)
    submission_id = uuid4().hex
    study_directory = machine.output_root / study.name
    records: list[JobRecord] = []
    error: str | None = None
    for group in groups:
        tasks = tuple(
            TaskRecord(
                index=task.index,
                grid=task.resolved.run.grid,
                run_path=task.resolved.run.run_path,
                command=task.command,
                extra_args=task.extra_args,
                resources=(
                    dataclasses.asdict(task.resolved.slurm)
                    if task.resolved.slurm is not None
                    else {}
                ),
                attempt=task.attempt,
                progress=take_progress(run_directory=study_directory / task.resolved.run.run_path),
            )
            for task in group.tasks
        )
        try:
            job_id = launcher.submit(group=group, study_folder=folder)
        except SlurmError as exc:
            error = str(exc)
            break
        record = JobRecord(
            kind=kind,
            submission_id=submission_id,
            time=utc_now_iso(),
            launcher=machine.launcher,
            job_id=job_id,
            job_file=group.job_file,
            tasks=tasks,
        )
        records.append(record)
        append_job(path=folder / 'submissions.jsonl', record=record)
    return LaunchReport(
        submitted=tuple(records), not_submitted=tuple(groups[len(records) :]), error=error
    )

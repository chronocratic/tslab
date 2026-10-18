"""Submitting planned job groups: the job files, and the launch itself."""

from collections.abc import Callable, Mapping, Sequence
import dataclasses
from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import subprocess
import time
from typing import Protocol, TextIO
from uuid import uuid4

from tslab.core.meta import hostname
from tslab.core.paths import atomic_write_text
from tslab.core.status import utc_now_iso
from tslab.study.definition import Study
from tslab.study.machine import MachineProfile, SlurmSettings
from tslab.study.plan import JobGroup, next_job_file, PlannedTask, study_folder
from tslab.study.records import append_job, JobRecord, take_progress, TaskRecord
from tslab.study.slurm import job_script, sbatch, SlurmError

__all__ = [
    'LaunchReport',
    'Launcher',
    'LocalLauncher',
    'SlurmLauncher',
    'launch',
    'submit_helper_job',
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


def write_job_files(
    *,
    study_folder: Path,
    groups: Sequence[JobGroup],
    machine: MachineProfile,
    project_root: Path,
    begin: str | None = None,
) -> None:
    """The ``jobs/<n>.tasks`` and ``jobs/<n>.sbatch`` files of every group.

    Args:
        study_folder: The ``.study`` folder of the study.
        groups: The planned groups.
        machine: The machine the jobs run on; its ``[env]`` goes in the scripts.
        project_root: The directory the scripts ``cd`` into.
        begin: A SLURM ``--begin`` for the jobs.
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
            begin=begin,
        )
        atomic_write_text(path=study_folder / 'jobs' / f'{group.job_file}.sbatch', text=script)


class Launcher(Protocol):
    """What one launch does with a written group: returns the job id."""

    def submit(self, *, group: JobGroup, study_folder: Path, record: Callable[[str], None]) -> str:
        """Submit the job files of one group and return the job id.

        Args:
            group: The group to submit.
            study_folder: The ``.study`` folder of the study.
            record: Called with the job id as soon as the job exists, before
                any task of it has run, so the record outlives an interrupted launch.
        """


class SlurmLauncher:
    """One ``sbatch`` per group, paused between the limits of the cluster."""

    def __init__(self, *, sleep: Callable[[float], None] = time.sleep) -> None:
        """Set up the launcher.

        Args:
            sleep: What waits out the submission pause of the cluster.
        """
        self._sleep = sleep

    def submit(self, *, group: JobGroup, study_folder: Path, record: Callable[[str], None]) -> str:
        """The id of the ``sbatch`` job of the group."""
        job_id = sbatch(
            script=study_folder / 'jobs' / f'{group.job_file}.sbatch', sleep=self._sleep
        )
        record(job_id)
        if group.slurm is not None:
            self._sleep(group.slurm.submit_pause_seconds)
        return job_id


class LocalLauncher:
    """The tasks of the group, one after another, in the foreground."""

    def __init__(self, *, project_root: Path, env: Mapping[str, str]) -> None:
        """Set up the launcher.

        Args:
            project_root: The directory the tasks run in.
            env: The ``[env]`` of the machine.
        """
        self._project_root = project_root
        self._env = env

    def submit(
        self,
        *,
        group: JobGroup,
        study_folder: Path,  # noqa: ARG002 # why: the Protocol signature
        record: Callable[[str], None],
    ) -> str:
        """Run every task, continuing after a failing one; the job id.

        The job is recorded before the first task starts. Each task appends
        to its log file (header, then the output line by line, which the
        console also gets); a task that exits non-zero or cannot start is
        logged without stopping the group.
        """
        job_id = f'local-{os.getpid()}-{group.job_file}'
        record(job_id)
        for task in group.tasks:
            task.log_file.parent.mkdir(parents=True, exist_ok=True)
            header = (
                f'=== tslab attempt {task.attempt} | job {job_id} '
                f'| {hostname()} | {utc_now_iso()} ===\n'
            )
            with task.log_file.open('a', encoding='utf-8') as log:
                log.write(header)
                log.flush()
                try:
                    exit_code = self._run_task(task=task, log=log)
                except OSError as error:
                    exit_code = None
                    line = f'=== task {task.index} could not start: {error} ===\n'
                    print(line, end='')  # noqa: T201 # why: the console output is the design
                    log.write(line)
            if exit_code is not None and exit_code != 0:
                line = f'=== task {task.index} exited with {exit_code} ===\n'
                print(line, end='')  # noqa: T201 # why: the console output is the design
                with task.log_file.open('a', encoding='utf-8') as log:
                    log.write(line)
        return job_id

    def _run_task(self, *, task: PlannedTask, log: TextIO) -> int:
        process = subprocess.Popen(  # noqa: S603 # why: the run command, typed by the study
            list(task.command),  # why: its first word is the user's script command
            cwd=self._project_root,
            env={**os.environ, **self._env},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
        )
        stdout = process.stdout
        if stdout is None:  # pragma: no cover — PIPE always has a stream
            message = f'no output stream for task {task.index}'
            raise RuntimeError(message)
        for line in stdout:
            print(line, end='')  # noqa: T201 # why: the console output is the design
            log.write(line)
        log.flush()
        return process.wait()


def launch(  # noqa: PLR0913 # why: keyword-only public API; each option is documented below
    *,
    launcher: Launcher,
    machine: MachineProfile,
    study: Study,
    groups: Sequence[JobGroup],
    project_root: Path,
    dry_run: bool,
    begin: str | None = None,
) -> LaunchReport:
    """Write the job files, then submit the groups one by one.

    A dry run writes and stops. Otherwise each group is recorded in
    ``submissions.jsonl`` as soon as its job exists and before the next is
    submitted, so a failing submission leaves exactly the records of the
    jobs that were sent; the launch stops there and reports the rest as
    not submitted.

    Args:
        launcher: What submits the written groups.
        machine: The machine the jobs run on.
        study: The study the groups came from.
        groups: The planned groups, in submission order.
        project_root: The directory the jobs run in.
        dry_run: Write the job files and stop, submitting nothing.
        begin: A SLURM ``--begin`` for the jobs.

    Returns:
        What was submitted, what was not, and the error that stopped the launch.
    """
    folder = study_folder(machine=machine, study=study)
    write_job_files(
        study_folder=folder, groups=groups, machine=machine, project_root=project_root, begin=begin
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

        def record_job(
            job_id: str, group: JobGroup = group, tasks: tuple[TaskRecord, ...] = tasks
        ) -> None:
            record = JobRecord(
                kind='run',
                submission_id=submission_id,
                time=utc_now_iso(),
                launcher=machine.launcher,
                job_id=job_id,
                job_file=group.job_file,
                tasks=tasks,
            )
            records.append(record)
            append_job(path=folder / 'submissions.jsonl', record=record)

        try:
            launcher.submit(group=group, study_folder=folder, record=record_job)
        except SlurmError as exc:
            error = str(exc)
            break
    return LaunchReport(
        submitted=tuple(records), not_submitted=tuple(groups[len(records) :]), error=error
    )


def submit_helper_job(  # noqa: PLR0913 # why: keyword-only public API; each option is documented below
    *,
    study_folder: Path,
    job_name: str,
    command: str,
    slurm: SlurmSettings,
    machine: MachineProfile,
    project_root: Path,
    dependency: str,
    sleep: Callable[[float], None] = time.sleep,
) -> JobRecord:
    """Write, submit and record one single-task helper job, e.g. a follow-up.

    Args:
        study_folder: The ``.study`` folder of the study.
        job_name: The job name; it also names the log file.
        command: The shell-quoted command of the job.
        slurm: The resources of the job.
        machine: The machine the job runs on; its ``[env]`` goes in the script.
        project_root: The directory the script ``cd``s into.
        dependency: The SLURM ``--dependency`` of the job.
        sleep: What waits out the submission pause of the cluster.

    Returns:
        The record of the submitted job, already appended to the study.

    Raises:
        SlurmError: If ``sbatch`` fails; nothing is recorded then.
    """
    job_file = next_job_file(study_folder=study_folder)
    (study_folder / 'jobs').mkdir(parents=True, exist_ok=True)
    (study_folder / 'logs').mkdir(parents=True, exist_ok=True)
    tasks_file = study_folder / 'jobs' / f'{job_file}.tasks'
    sbatch_file = study_folder / 'jobs' / f'{job_file}.sbatch'
    atomic_write_text(path=tasks_file, text=f'0 1 {command}\n')
    atomic_write_text(
        path=sbatch_file,
        text=job_script(
            job_name=job_name,
            slurm=slurm,
            indices=None,
            tasks_file=tasks_file,
            log_pattern=study_folder / 'logs' / f'followup_{job_file}.out',
            project_root=project_root,
            env=machine.env,
            dependency=dependency,
        ),
    )
    job_id = sbatch(script=sbatch_file, sleep=sleep)
    record = JobRecord(
        kind='followup',
        submission_id=uuid4().hex,
        time=utc_now_iso(),
        launcher='slurm',
        job_id=job_id,
        job_file=job_file,
        tasks=(),
    )
    append_job(path=study_folder / 'submissions.jsonl', record=record)
    return record

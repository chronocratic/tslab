"""Submitting planned job groups: the job files, and the launch itself."""

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import logging
import os
from pathlib import Path
import shlex
import signal
import sys
import time
from typing import cast, Protocol, TextIO
from uuid import uuid4

from tslab.core.files import atomic_write_text
from tslab.core.meta import hostname
from tslab.core.status import utc_now_iso
from tslab.jobs.local import (
    gpu_lock_folder,
    PoolCommand,
    run_commands,
    select_gpus,
    stop_on_signals,
    TaskInfo,
    TaskOutput,
)
from tslab.jobs.slurm import sbatch, SlurmError
from tslab.study.definition import Study
from tslab.study.job_script import job_script
from tslab.study.launchers import LaunchReport, LaunchStopped, OnTaskState
from tslab.study.machine import LocalSettings, MachineProfile, SlurmSettings, timeout_seconds
from tslab.study.plan import JobGroup, next_job_file, study_folder
from tslab.study.records import append_job, JobRecord, take_progress, TaskRecord

__all__ = [
    'SIGNAL_EXIT_BASE',
    'Launcher',
    'LocalLauncher',
    'SlurmLauncher',
    'launch',
    'local_job_id',
    'resolve_local_gpus',
    'submit_helper_job',
    'usage_file',
    'write_job_files',
]

_LOGGER = logging.getLogger(__name__)

#: The base of an exit code made from a signal: the code is the base plus the number.
SIGNAL_EXIT_BASE = 128

#: The signals that stop a local launch; the records of the groups sent survive.
_STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


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
        if group.resources.max_tasks_per_job is None:
            continue
        script = job_script(
            job_name=group.job_name,
            slurm=cast('SlurmSettings', group.resources),
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
        if group.resources.max_tasks_per_job is not None:
            slurm = cast('SlurmSettings', group.resources)
            self._sleep(slurm.submit_pause_seconds)
        return job_id


def local_job_id(*, job_file: int) -> str:
    """The job id of the local launch of one group file.

    Args:
        job_file: The group file number of the launch.
    """
    return f'local-{os.getpid()}-{job_file}'


def usage_file(*, log_file: Path) -> Path:
    """The usage file written next to the log of one local task.

    Args:
        log_file: The log file of the task.
    """
    return log_file.with_name(log_file.name + '.usage.json')


def resolve_local_gpus(*, local: LocalSettings, environment: Mapping[str, str]) -> tuple[str, ...]:
    """The GPU ids of a local launch, with ``"auto"`` resolved and the busy ones left out.

    Args:
        local: The ``[local]`` table: the ``gpus`` list and ``skip_busy_gpus``.
        environment: The environment to discover the GPUs from.

    Returns:
        The ids the launch may lease, in order; empty for CPU-only slots.

    Raises:
        ValueError: As ``select_gpus``.
    """
    gpus = 'auto' if local.gpus == 'auto' else tuple(str(gpu) for gpu in local.gpus)
    gpu_ids = select_gpus(gpus=gpus, skip_busy=local.skip_busy_gpus, environment=environment)
    if gpu_ids and local.max_parallel != 1:
        _LOGGER.warning('local.gpus is set, so local.max_parallel is ignored')
    if not gpu_ids and local.tasks_per_gpu > 1:
        _LOGGER.warning('no local GPUs, so local.tasks_per_gpu is ignored')
    return gpu_ids


class LocalLauncher:
    """The tasks of the group in parallel slots, under a per-task time limit.

    Without GPUs at most ``local.max_parallel`` tasks run at once; with GPUs
    one slot per (GPU, lane), the GPUs locked across launches in
    ``gpu_lock_folder``.
    """

    def __init__(  # noqa: PLR0913 # why: keyword-only public API; each option is documented below
        self,
        *,
        project_root: Path,
        env: Mapping[str, str],
        local: LocalSettings,
        gpu_ids: tuple[str, ...] = (),
        stream: TextIO | None = None,
        on_task_state: OnTaskState | None = None,
        grace_seconds: float = 30.0,
    ) -> None:
        """Set up the launcher.

        Args:
            project_root: The directory the tasks run in.
            env: The ``[env]`` of the machine.
            local: The ``[local]`` table: time limit, signals, parallelism.
            gpu_ids: The resolved GPU ids the tasks lease; the caller
                resolves ``"auto"``. Empty for CPU-only slots.
            stream: Console output stream; standard output when omitted.
            on_task_state: Called with the task id and ``'running'`` or
                ``'ended'``; the task id is the job id and the task index.
            grace_seconds: Seconds to wait after a stop's ``SIGTERM`` before
                a ``SIGKILL``.
        """
        self._project_root = project_root
        self._env = env
        self._local = local
        self._gpu_ids = gpu_ids
        self._stream = stream
        self._on_task_state = on_task_state
        self._grace_seconds = grace_seconds

    def submit(
        self,
        *,
        group: JobGroup,
        study_folder: Path,  # noqa: ARG002 # why: the Protocol signature
        record: Callable[[str], None],
    ) -> str:
        """Run every task in slots, continuing after a failing one; the job id.

        The job is recorded before the first task starts. Each task appends
        to its log file (header, then the output line by line); with one
        slot the console gets the lines too, with more slots it gets one
        line per start and per end. After each task the launcher writes the
        usage file and a footer line to the log; a task that exits
        non-zero or cannot start does not stop the group.

        A SIGINT, SIGTERM or SIGHUP to this process stops the pool: running
        tasks get a SIGTERM, then a SIGKILL after ``grace_seconds``, and no
        task starts after the signal.

        Raises:
            LaunchStopped: If a stop signal was caught, with its number.
        """
        job_id = local_job_id(job_file=group.job_file)
        record(job_id)
        stream = sys.stdout if self._stream is None else self._stream
        if self._gpu_ids:
            slots = len(self._gpu_ids) * self._local.tasks_per_gpu
        else:
            slots = self._local.max_parallel
        by_key = {f'{job_id}_{task.index}': task for task in group.tasks}

        def task_header(key: str) -> str:
            task = by_key[key]
            header = (
                f'=== tslab attempt {task.attempt} | job {job_id} | {hostname()} | {utc_now_iso()}'
            )
            if self._gpu_ids:
                header += f' | gpus {",".join(self._gpu_ids)}'
            return header + ' ===\n'

        def on_task_state(key: str, state: str) -> None:
            if self._on_task_state is not None:
                self._on_task_state(task_id=key, state=state)

        output = TaskOutput(
            stream=stream,
            single_slot=slots == 1,
            tasks={
                key: TaskInfo(
                    log_file=task.log_file,
                    index=task.index,
                    name=f'run {task.resolved.run.run_path}',
                    usage_file=usage_file(log_file=task.log_file),
                    usage_document={
                        'schema': 'tslab.local-usage/1',
                        'attempt': task.attempt,
                        'job_id': job_id,
                        'timeout': timeout_seconds(local=self._local),
                    },
                )
                for key, task in by_key.items()
            },
            header=task_header,
            on_task_state=on_task_state,
        )
        with stop_on_signals(signals=_STOP_SIGNALS) as request:
            run_commands(
                commands=[
                    PoolCommand(
                        key=key,
                        command=task.command,
                        cwd=self._project_root,
                        environment={**os.environ, **self._env},
                    )
                    for key, task in by_key.items()
                ],
                slots=slots,
                gpu_ids=self._gpu_ids,
                tasks_per_gpu=self._local.tasks_per_gpu,
                lock_folder=(
                    gpu_lock_folder(name='tslab-gpus', environment=os.environ)
                    if self._gpu_ids
                    else None
                ),
                stop=request.event,
                timeout_seconds=timeout_seconds(local=self._local),
                signal_seconds=self._local.signal_seconds,
                on_start=output.on_start,
                on_line=output.on_line,
                on_event=output.on_event,
                on_wait=output.on_wait,
                on_end=output.on_end,
                grace_seconds=self._grace_seconds,
            )
        if request.signal_number is not None:
            raise LaunchStopped(signal_number=request.signal_number)
        return job_id


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
        What was submitted, what was not, the error that stopped the launch,
        and the signal that stopped it.
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
    stopped_signal: int | None = None
    for group in groups:
        tasks = tuple(
            TaskRecord(
                index=task.index,
                grid=task.resolved.run.grid,
                run_path=task.resolved.run.run_path,
                command=task.command,
                extra_args=task.extra_args,
                resources=(
                    dataclasses.asdict(cast('SlurmSettings', task.resolved.resources))
                    if task.resolved.resources.max_tasks_per_job is not None
                    else {}
                ),
                attempt=task.attempt,
                progress=take_progress(run_directory=study_directory / task.resolved.run.run_path),
                uncounted_attempts=task.uncounted_attempts,
                uncounted_in_row=task.uncounted_in_row,
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
        except LaunchStopped as exc:
            # why: the record of this group is kept; resubmit continues the stopped runs
            stopped_signal = exc.signal_number
            break
    return LaunchReport(
        submitted=tuple(records),
        not_submitted=tuple(groups[len(records) :]),
        error=error,
        stopped_signal=stopped_signal,
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

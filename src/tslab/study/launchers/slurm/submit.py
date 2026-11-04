"""Writing and submitting SLURM job files."""

from collections.abc import Callable, Sequence
import os
from pathlib import Path
import shlex
import time
from typing import cast
from uuid import uuid4

from tslab.core.files import atomic_write_text
from tslab.enums import LauncherKind
from tslab.jobs.slurm import sbatch, SlurmError
from tslab.study.code_snapshot import without_code_snapshot
from tslab.study.launchers.common.protocols import RecordJob
from tslab.study.launchers.common.types import LaunchError
from tslab.study.launchers.slurm.job_script import job_script
from tslab.study.launchers.slurm.records import SlurmJobRecord
from tslab.study.machine import MachineProfile, SlurmSettings
from tslab.study.paths import next_job_file
from tslab.study.plan import JobGroup
from tslab.study.records import append_job, JobKind

__all__ = ['SlurmLauncher', 'submit_helper_job', 'write_job_files']


def write_job_files(
    *,
    study_folder: Path,
    groups: Sequence[JobGroup],
    machine: MachineProfile,
    project_root: Path,
    start_delay_seconds: int | None = None,
) -> None:
    """The ``jobs/<n>.tasks`` and ``jobs/<n>.sbatch`` files of every group.

    Args:
        study_folder: The ``.study`` folder of the study.
        groups: The planned groups.
        machine: The machine the jobs run on; its ``[env]`` goes in the scripts.
        project_root: The directory the scripts ``cd`` into.
        start_delay_seconds: Seconds the jobs wait before they may start; written as a ``--begin``.
    """
    begin = None if start_delay_seconds is None else f'now+{start_delay_seconds}'
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
            log_pattern=_log_pattern(group=group),
            project_root=project_root,
            env=machine.env,
            begin=begin,
            code_snapshot=group.code_snapshot,
        )
        atomic_write_text(path=study_folder / 'jobs' / f'{group.job_file}.sbatch', text=script)


def _log_pattern(*, group: JobGroup) -> Path:
    """The ``--output`` pattern of a group: the task log files with the index as ``%a``."""
    first = group.tasks[0]
    suffix = f'_{first.index}.out'
    if not first.log_file.name.endswith(suffix):
        message = (
            f'cannot derive the log pattern from {first.log_file.name!r}; expected ...{suffix}'
        )
        raise ValueError(message)
    stem = first.log_file.name.removesuffix(suffix)
    return first.log_file.with_name(f'{stem}_%a.out')


class SlurmLauncher:
    """One ``sbatch`` per group, paused between the limits of the cluster."""

    def __init__(self, *, sleep: Callable[[float], None] = time.sleep) -> None:
        """Set up the launcher.

        Args:
            sleep: What waits out the submission pause of the cluster.
        """
        self._sleep = sleep

    def submit(self, *, group: JobGroup, study_folder: Path, record: RecordJob) -> str:
        """The id of the ``sbatch`` job of the group.

        Raises:
            LaunchError: If ``sbatch`` fails; nothing is recorded then.
        """
        try:
            # why: a nested launch must not submit with the snapshot it itself runs from
            job_id = sbatch(
                script=study_folder / 'jobs' / f'{group.job_file}.sbatch',
                sleep=self._sleep,
                environ=without_code_snapshot(environ=os.environ),
            )
        except SlurmError as error:
            raise LaunchError(str(error)) from error
        record(job_id=job_id)
        if group.resources.max_tasks_per_job is not None:
            slurm = cast('SlurmSettings', group.resources)
            self._sleep(slurm.submit_pause_seconds)
        return job_id


def submit_helper_job(  # noqa: PLR0913 # why: keyword-only public API; each option is documented below
    *,
    study_folder: Path,
    job_name: str,
    command: str,
    slurm: SlurmSettings,
    machine: MachineProfile,
    project_root: Path,
    dependency: str,
    now: Callable[[], str],
    sleep: Callable[[float], None] = time.sleep,
) -> SlurmJobRecord:
    """Write, submit and record one single-task helper job, e.g. a follow-up.

    Args:
        study_folder: The ``.study`` folder of the study.
        job_name: The job name; it also names the log file.
        command: The shell-quoted command of the job.
        slurm: The resources of the job.
        machine: The machine the job runs on; its ``[env]`` goes in the script.
        project_root: The directory the script ``cd``s into.
        dependency: The SLURM ``--dependency`` of the job.
        now: Return the current UTC ISO timestamp for the record.
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
    job_id = sbatch(
        script=sbatch_file, sleep=sleep, environ=without_code_snapshot(environ=os.environ)
    )
    record = SlurmJobRecord(
        kind=JobKind.FOLLOWUP,
        submission_id=uuid4().hex,
        time=now(),
        launcher=LauncherKind.SLURM,
        job_id=job_id,
        job_file=job_file,
        tasks=(),
        job_name=job_name,
    )
    append_job(path=study_folder / 'submissions.jsonl', record=record)
    return record

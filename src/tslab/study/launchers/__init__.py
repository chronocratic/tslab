"""The launch backends: one object per launcher, SLURM and local apart.

A study command asks the backend of the machine it runs on, picked once by
``backend_for`` from the profile's launcher; it never names a launcher. The
backend code lives in the ``slurm`` and ``local`` subpackages, which never
import each other.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum
import os
from pathlib import Path
import sys
from typing import Protocol, TextIO
from uuid import uuid4

from tslab.study.definition import Study
from tslab.study.doctor import DoctorCheck
from tslab.study.machine import MachineOverride, MachineProfile
from tslab.study.paths import study_folder
from tslab.study.plan import JobGroup
from tslab.study.records import append_job, JobRecord, take_progress, TaskRecord
from tslab.study.reporting.state import RunReport

__all__ = [
    'BACKENDS',
    'SIGNAL_EXIT_BASE',
    'Backend',
    'JobFlag',
    'JobPhase',
    'JobStates',
    'JobUsage',
    'LaunchError',
    'LaunchReport',
    'LaunchStopped',
    'Launcher',
    'OnLaunch',
    'OnTaskState',
    'QueuedJob',
    'RecordJob',
    'Resources',
    'available_cpus',
    'backend_for',
    'launch',
]

#: The base of an exit code made from a signal: the code is the base plus the number.
SIGNAL_EXIT_BASE = 128


class LaunchError(RuntimeError):
    """The launch could not be executed on the machine; the message is the failure."""


class LaunchStopped(RuntimeError):  # noqa: N818 # why: names the stop event, not an error
    """A signal stopped the launch before every group was submitted.

    Attributes:
        signal_number: The number of the first signal that stopped the launch.
    """

    def __init__(self, *, signal_number: int) -> None:
        """Set the number of the first signal that stopped the launch."""
        message = f'stopped by signal {signal_number}'
        super().__init__(message)
        self.signal_number = signal_number


@dataclass(frozen=True, kw_only=True)
class LaunchReport:
    """What one launch submitted.

    Args:
        submitted: One job record per submitted group, in order.
        not_submitted: The groups the launch did not reach; all of them
            on a dry run, the rest after an error or a signal.
        error: The submission error that stopped the launch, if any.
        stopped_signal: The signal that stopped the launch, if any.
    """

    submitted: tuple[JobRecord, ...]
    not_submitted: tuple[JobGroup, ...]
    error: str | None
    stopped_signal: int | None = None


class OnTaskState(Protocol):
    """The state changes of the tasks of one local launch."""

    def __call__(self, *, task_id: str, state: str) -> None:
        """Called with the id of a task and its new state."""
        ...


class OnLaunch(Protocol):
    """Called once with the result of a finished launch, dry runs included."""

    def __call__(self, *, report: LaunchReport, groups: Sequence[JobGroup]) -> None:
        """Called with the launch report and the groups the launch covered."""
        ...


class JobPhase(StrEnum):
    """The phase of a job state; the scheduler's words stay in ``JobStates.states``.

    ``ENDED`` is over with the cause unknown; ``UNKNOWN`` was not queried.
    """

    QUEUED = 'queued'
    RUNNING = 'running'
    INTERRUPTED = 'interrupted'
    FAILED = 'failed'
    COMPLETED = 'completed'
    ENDED = 'ended'
    UNKNOWN = 'unknown'


class JobFlag(StrEnum):
    """What a backend says a job state means for the state logic."""

    #: The job stopped at its time limit.
    TIME_LIMIT = 'time_limit'
    #: The machine, not the run, ended the job; the attempt does not count toward the cap.
    NOT_COUNTED = 'not_counted'


@dataclass(frozen=True, kw_only=True)
class JobStates:
    """The states of the jobs of one study, as one backend reports them.

    Args:
        known: Whether the states are real, or the scheduler could not be
            asked and the state logic works from the run folders only.
        states: The raw state text per job or task id; shown to the user.
        phases: The phase per id; what the state logic reads.
        reasons: The one-line reason per id, shown when the run folder gives none.
        flags: The flags per id; only ids that have a flag.
        followup_live_job_ids: The ids of the queued or running follow-up jobs.
    """

    known: bool
    states: Mapping[str, str]
    phases: Mapping[str, JobPhase]
    reasons: Mapping[str, str]
    flags: Mapping[str, frozenset[JobFlag]]
    followup_live_job_ids: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class JobUsage:
    """What the accounting of one task of a job reports, whatever the scheduler.

    Attributes:
        state: The first word of the task state, such as ``COMPLETED``.
        elapsed: The wall-clock time of the task, or ``None``.
        time_limit: The time limit of the task, or ``None``.
        total_cpu: The CPU seconds used, or ``None``.
        max_rss: The peak memory in bytes over the steps, or ``None``.
        alloc_cpus: The allocated CPUs, or ``None``.
        gpus: The GPUs of the allocation, else ``0``.
    """

    state: str
    elapsed: timedelta | None
    time_limit: timedelta | None
    total_cpu: timedelta | None
    max_rss: int | None
    alloc_cpus: int | None
    gpus: int


@dataclass(frozen=True, kw_only=True)
class QueuedJob:
    """What the scheduler reports about one queued job.

    Attributes:
        start: The expected start time, or ``None`` when the scheduler knows none.
        reason: The pending reason, or ``None`` when the scheduler gives none.
        stuck: Whether the reason will not clear on its own.
    """

    start: str | None
    reason: str | None
    stuck: bool


class Resources(Protocol):
    """The resources one launch needs, whatever the launcher is."""

    @property
    def cpus(self) -> int:
        """The CPUs one task of the launch gets."""
        ...

    @property
    def max_tasks_per_job(self) -> int | None:
        """The maximum number of tasks in one job, when the launcher has one."""
        ...

    @property
    def requested_memory(self) -> int | None:
        """The total memory in bytes one task requests, when the launcher requests any."""
        ...

    def describe(self) -> str:
        """The resource line of the settings, for the dry run printout."""
        ...

    def as_record(self) -> dict[str, object]:
        """The resource fields of the group, for the submission record."""
        ...

    def signal_problem(self) -> str | None:
        """The time-limit signal problem for one group.

        Returns:
            The problem message, or ``None`` when the timing is valid.
        """
        ...


# why: the types above must exist first: this import reaches
# the local pid-file module, which imports OnTaskState from this door
from tslab.study.context import Context  # noqa: E402


class RecordJob(Protocol):
    """Records the job id of one submitted group before its tasks run."""

    def __call__(self, *, job_id: str) -> None:
        """Record the id of the submitted group."""


class Launcher(Protocol):
    """What one launch does with a written group: returns the job id."""

    def submit(self, *, group: JobGroup, study_folder: Path, record: RecordJob) -> str:
        """Submit one group and record its job id before tasks run."""
        ...


class WriteJobFiles(Protocol):
    """Write the task and launcher files of a planned launch."""

    def __call__(
        self,
        *,
        study_folder: Path,
        groups: Sequence[JobGroup],
        machine: MachineProfile,
        project_root: Path,
        start_delay_seconds: int | None = None,
    ) -> None:
        """Write files needed by one backend's launch."""
        ...


def launch(  # noqa: PLR0913 # why: keyword-only public API; each launch input is explicit
    *,
    launcher: Launcher,
    write_job_files: WriteJobFiles,
    machine: MachineProfile,
    study: Study,
    groups: Sequence[JobGroup],
    project_root: Path,
    dry_run: bool,
    now: Callable[[], str],
    start_delay_seconds: int | None = None,
) -> LaunchReport:
    """Write backend files, then submit groups one by one.

    A dry run writes files and stops. Otherwise each group is recorded as soon
    as its job exists, so a failed launch keeps records for submitted groups.

    Args:
        launcher: The backend's implementation that submits one group.
        write_job_files: The backend's file writer.
        machine: The machine the jobs run on.
        study: The study the groups came from.
        groups: Planned groups in submission order.
        project_root: The directory the jobs run in.
        dry_run: Write files and stop without submitting.
        now: Return the current UTC ISO timestamp for each submitted group.
        start_delay_seconds: Seconds the jobs wait before they may start, when set.

    Returns:
        What was submitted, what was not, and any error or stopping signal.
    """
    folder = study_folder(machine=machine, study=study)
    write_job_files(
        study_folder=folder,
        groups=groups,
        machine=machine,
        project_root=project_root,
        start_delay_seconds=start_delay_seconds,
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
                resources=task.resolved.resources.as_record(),
                attempt=task.attempt,
                progress=take_progress(run_directory=study_directory / task.resolved.run.run_path),
                uncounted_attempts=task.uncounted_attempts,
                uncounted_in_row=task.uncounted_in_row,
            )
            for task in group.tasks
        )

        def record_job(
            *, job_id: str, group: JobGroup = group, tasks: tuple[TaskRecord, ...] = tasks
        ) -> None:
            record = JobRecord(
                kind='run',
                submission_id=submission_id,
                time=now(),
                launcher=machine.launcher,
                job_id=job_id,
                job_file=group.job_file,
                tasks=tasks,
                code_snapshot=group.code_snapshot,
            )
            records.append(record)
            append_job(path=folder / 'submissions.jsonl', record=record)

        try:
            launcher.submit(group=group, study_folder=folder, record=record_job)
        except LaunchError as exc:
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


class Backend(Protocol):
    """What one launcher does for a study: its states, resources and launch.

    A command picks the backend of its machine once, through ``backend_for``,
    and calls these methods; it never compares launcher names.
    """

    name: str
    #: Whether the jobs have a time limit; a missing one shows as ``-``, else as ``no limit``.
    has_time_limit: bool
    #: Whether an ended job state also covers jobs that ran on other hosts.
    knows_other_hosts: bool

    def job_states(
        self,
        *,
        machine: MachineProfile,
        study: Study,
        jobs: Sequence[JobRecord],
        cache: Mapping[str, str],
    ) -> tuple[JobStates, Mapping[str, str]]:
        """The states of the jobs, with the new cache of final states second."""
        ...

    def expected_starts(self, *, context: Context) -> Mapping[str, QueuedJob] | None:
        """The expected start and pending reason of the queued jobs.

        ``None`` without a scheduler.
        """
        ...

    def launch_blocker(self, *, context: Context) -> str | None:
        """Why a new launch of this study cannot start now, or ``None``."""
        ...

    def resources(
        self,
        *,
        machine: MachineProfile,
        overrides: Sequence[MachineOverride],
        grid_name: str,
        values: Mapping[str, str],
        local_cpus: int,
    ) -> Resources:
        """The resources of one grid, with the machine overrides applied."""
        ...

    def failure_hint(
        self, *, report: RunReport, usage: JobUsage | None, resources: Resources
    ) -> str | None:
        """The hint the scheduler state of a failed run gives; ``None`` when it has none."""
        ...

    def suggest_sizing(
        self, *, elapsed: timedelta | None, peak_per_cpu: int | None
    ) -> tuple[str | None, str | None]:
        """The suggested time limit and memory per CPU, from the past usage with margin.

        A backend without a time limit or memory request returns ``(None, None)``.
        """
        ...

    def foreground_script(
        self, *, context: Context, resources: Resources, command: Sequence[str]
    ) -> str:
        """The shell script that runs one study run in the terminal."""
        ...

    def launch(  # noqa: PLR0913 # why: the Backend signature; one keyword per launch input
        self,
        *,
        context: Context,
        groups: Sequence[JobGroup],
        folder: Path,
        dry_run: bool,
        start_delay_seconds: int | None,
        sleep: Callable[[float], None],
        task_stream: TextIO | None,
        environment: Mapping[str, str],
        pid: int,
        host: str,
        now: Callable[[], str],
    ) -> LaunchReport | None:
        """Submit the groups; ``None`` when a problem was logged, not raised."""
        ...

    def after_submit(
        self,
        *,
        context: Context,
        submitted: Sequence[JobRecord],
        retry: Callable[..., int],
        sleep: Callable[[float], None],
    ) -> int:
        """The follow-up of a finished launch, as an exit code."""
        ...

    def cancel(
        self, *, context: Context, dry_run: bool, signal: str | None
    ) -> tuple[int, list[str]]:
        """Cancel the live jobs; the exit code and the touched job ids."""
        ...

    def cancellable_job_ids(self, *, reports: Sequence[RunReport]) -> list[str]:
        """The job ids a cancel can reach for the reports."""
        ...

    def detach(self, *, context: Context, arguments: Sequence[str], wait_command: str) -> int:
        """Start the launch in the background; the exit code."""
        ...

    def failure_usage(
        self, *, context: Context, reports: Sequence[RunReport]
    ) -> Mapping[str, JobUsage]:
        """The accounting usage of the failed runs, when the scheduler gives it."""
        ...

    def usage(self, *, context: Context) -> tuple[Mapping[str, JobUsage], Mapping[str, str | None]]:
        """The usage by job id, and the GPU id by job id."""
        ...

    def doctor_checks(self) -> tuple[DoctorCheck, ...]:
        """The readiness checks of the machines the backend serves."""
        ...


# why: the backends import the types from this door
from tslab.study.launchers.local import LocalBackend  # noqa: E402
from tslab.study.launchers.local.resources import usable_cpus  # noqa: E402
from tslab.study.launchers.slurm import SlurmBackend  # noqa: E402

#: One backend per launcher; the only place beside the machine tables that
#: names both.
BACKENDS: Mapping[str, Backend] = {'slurm': SlurmBackend(), 'local': LocalBackend()}


def backend_for(*, machine: MachineProfile) -> Backend:
    """The backend that serves the machine's launcher.

    Args:
        machine: The machine profile to read the launcher from.

    Returns:
        The backend object of the machine's launcher.
    """
    backend = BACKENDS.get(machine.launcher)
    if backend is None:
        message = f'no backend serves launcher {machine.launcher!r}; check the machine profile'
        raise ValueError(message)
    return backend


def available_cpus() -> int:
    """The CPUs this process may run on; a local launch runs its tasks here."""
    # why: the affinity mask honours taskset and container limits; macOS has no such call
    if sys.platform == 'linux':
        return usable_cpus(affinity=lambda: len(os.sched_getaffinity(0)))
    return usable_cpus(affinity=lambda: os.cpu_count() or 1)

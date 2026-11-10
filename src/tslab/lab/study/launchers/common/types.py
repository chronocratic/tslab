"""The types a launch and its backends exchange: errors, reports, states and callbacks."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from types import MappingProxyType
from typing import Protocol

from tslab.lab.enums import JobFlag, JobPhase, LocalTaskState
from tslab.lab.study.plan import JobGroup
from tslab.lab.study.records import JobRecord

__all__ = [
    'SIGNAL_EXIT_BASE',
    'JobStates',
    'JobUsage',
    'LaunchError',
    'LaunchReport',
    'LaunchStopped',
    'OnLaunch',
    'OnTaskState',
    'QueuedJob',
    'Resources',
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

    def __call__(self, *, task_id: str, state: LocalTaskState, pid: int | None = None) -> None:
        """Called with the id of a task, its new state, and its pid while running."""
        ...


class OnLaunch(Protocol):
    """Called once with the result of a finished launch, dry runs included."""

    def __call__(self, *, report: LaunchReport, groups: Sequence[JobGroup]) -> None:
        """Called with the launch report and the groups the launch covered."""
        ...


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
        start_times: The UTC start time in Unix seconds, when known.
    """

    known: bool
    states: Mapping[str, str]
    phases: Mapping[str, JobPhase]
    reasons: Mapping[str, str]
    flags: Mapping[str, frozenset[JobFlag]]
    followup_live_job_ids: tuple[str, ...]
    start_times: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))


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
        exit_code: The exit code the scheduler recorded for the job, or ``None``.
    """

    state: str
    elapsed: timedelta | None
    time_limit: timedelta | None
    total_cpu: timedelta | None
    max_rss: int | None
    alloc_cpus: int | None
    gpus: int
    exit_code: str | None = None


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
    def cpus_per_run(self) -> int:
        """The CPUs of one run when several run at once."""
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

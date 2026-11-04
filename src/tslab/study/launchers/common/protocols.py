"""The protocols each backend implements: its launcher, job files and backend object."""

from collections.abc import Callable, Mapping, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Protocol, TextIO

from tslab.enums import LauncherKind
from tslab.study.context import Context
from tslab.study.definition import Study
from tslab.study.doctor import DoctorCheck
from tslab.study.launchers.common.types import (
    JobStates,
    JobUsage,
    LaunchReport,
    QueuedJob,
    Resources,
)
from tslab.study.machine import MachineOverride, MachineProfile
from tslab.study.plan import JobGroup
from tslab.study.records import JobRecord
from tslab.study.reporting.state import RunReport

__all__ = ['Backend', 'Launcher', 'RecordJob', 'WriteJobFiles']


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


class Backend(Protocol):
    """What one launcher does for a study: its states, resources and launch.

    A command picks the backend of its machine once, through ``backend_for``,
    and calls these methods; it never compares launcher names.
    """

    name: LauncherKind
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

    def after_stopped_submit(self, *, context: Context, submitted: Sequence[JobRecord]) -> None:
        """The bookkeeping of a launch that a signal stopped; no follow-up is queued."""
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

    def doctor_checks(self, *, machine: MachineProfile) -> tuple[DoctorCheck, ...]:
        """The readiness checks of the machine profile the backend serves."""
        ...

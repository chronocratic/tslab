"""A fake backend for the study tests: recorded calls, canned answers, no SLURM.

``FakeBackend`` implements every member of ``tslab.study.launchers.Backend``
with neutral values. A test sets the answers it wants before the command
runs, and reads the calls the command made. ``install_fake_backend`` puts
the fake in place of both real backends, so a command that picks its
backend by machine profile reaches the fake, for ``slurm`` and ``local``.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
import shlex
from types import MappingProxyType
from typing import TextIO

import pytest

from tslab.core.enums import LauncherKind
from tslab.study.context import Context
from tslab.study.definition import Study
from tslab.study.doctor import DoctorCheck
import tslab.study.launchers
from tslab.study.launchers import (
    Backend,
    JobFlag,
    JobPhase,
    JobStates,
    JobUsage,
    launch,
    Launcher,
    LaunchReport,
    QueuedJob,
    RecordJob,
    Resources,
)
from tslab.study.machine import MachineOverride, MachineProfile
from tslab.study.plan import JobGroup
from tslab.study.records import JobRecord
from tslab.study.reporting.state import RunReport

__all__ = ['FakeBackend', 'FakeResources', 'install_fake_backend']

_EMPTY_STATES = JobStates(
    known=True,
    states=MappingProxyType({}),
    phases=MappingProxyType({}),
    reasons=MappingProxyType({}),
    flags=MappingProxyType({}),
    followup_live_job_ids=(),
)


@dataclass(frozen=True, kw_only=True)
class FakeResources:
    """The resources of a fake launch; small, and hashable for the planner.

    Attributes:
        cpus: The CPUs one task of the launch gets.
        cpus_per_run: The CPUs of one run when several run at once.
        max_tasks_per_job: The maximum number of tasks in one job, when set.
        requested_memory: The total memory in bytes one task requests, when set.
    """

    cpus: int = 1
    cpus_per_run: int = 1
    max_tasks_per_job: int | None = None
    requested_memory: int | None = None

    def describe(self) -> str:
        """The resource line of the settings, for the dry run printout."""
        return 'fake resources'

    def as_record(self) -> dict[str, object]:
        """The resource fields of the group, for the submission record."""
        return {}

    def signal_problem(self) -> str | None:
        """A fake launch has no time-limit signal problem."""
        return None


class _FakeLauncher:
    """Hands out the fake job ids, one per group, through the shared launch."""

    def __init__(self, *, next_job_id: Callable[[], str]) -> None:
        self._next_job_id = next_job_id

    def submit(self, *, group: JobGroup, study_folder: Path, record: RecordJob) -> str:
        """Submit one group and record its job id before its tasks run."""
        job_id = self._next_job_id()
        record(job_id=job_id)
        return job_id


def _write_no_job_files(
    *,
    study_folder: Path,
    groups: Sequence[JobGroup],
    machine: MachineProfile,
    project_root: Path,
    start_delay_seconds: int | None = None,
) -> None:
    """Write no launcher files; a fake launch needs none."""


class FakeBackend:
    """A backend for tests: every call is recorded, every answer is settable.

    The answers hold neutral values: job ids are ``job-0``, ``job-1``, and so
    on, state text is the phase word, and no method imitates a scheduler.

    Attributes:
        job_states_answer: The states ``job_states`` returns.
        resources_answer: The resources ``resources`` returns.
        expected_starts_answer: The starts ``expected_starts`` returns.
        after_submit_code: The exit code ``after_submit`` returns.
        cancel_answer: The exit code and ids ``cancel`` returns.
        cancellable_answer: The ids ``cancellable_job_ids`` returns.
        detach_code: The exit code ``detach`` returns.
        usage_answer: The usage ``usage`` returns.
        gpu_ids_answer: The GPU ids ``usage`` returns.
        failure_usage_answer: The usage ``failure_usage`` returns.
        doctor_answer: The checks ``doctor_checks`` returns.
        sizing_answer: The sizing ``suggest_sizing`` returns.
        failure_hint_answer: The hint ``failure_hint`` returns.
        errors: One exception per method name, raised when that method is
            called, so a test can make one call fail.
    """

    name = LauncherKind.LOCAL
    has_time_limit = True
    knows_other_hosts = True

    def __init__(self) -> None:
        self.job_states_answer: JobStates = _EMPTY_STATES
        self.resources_answer: Resources = FakeResources()
        self.expected_starts_answer: dict[str, QueuedJob] | None = None
        self.after_submit_code: int = 0
        self.cancel_answer: tuple[int, list[str]] = (0, [])
        self.cancellable_answer: list[str] = []
        self.detach_code: int = 0
        self.usage_answer: Mapping[str, JobUsage] = {}
        self.gpu_ids_answer: Mapping[str, str | None] = {}
        self.failure_usage_answer: Mapping[str, JobUsage] = {}
        self.doctor_answer: tuple[DoctorCheck, ...] = ()
        self.sizing_answer: tuple[str | None, str | None] = (None, None)
        self.failure_hint_answer: str | None = None
        self.errors: dict[str, Exception] = {}
        self._calls: list[tuple[str, dict[str, object]]] = []
        self._job_count: int = 0

    def _record(self, *, method: str, **kwargs: object) -> None:
        """Record the call, then raise the error the test set for the method, if any."""
        self._calls.append((method, dict(kwargs)))
        error = self.errors.get(method)
        if error is not None:
            raise error

    def _take_job_id(self) -> str:
        """The next fake job id, counting up across every launch of this instance."""
        job_id = f'job-{self._job_count}'
        self._job_count += 1
        return job_id

    def set_states(  # noqa: PLR0913 # why: one keyword per JobStates field
        self,
        *,
        phases: Mapping[str, JobPhase],
        states: Mapping[str, str] | None = None,
        reasons: Mapping[str, str] | None = None,
        flags: Mapping[str, frozenset[JobFlag]] | None = None,
        known: bool = True,
        followup_live_job_ids: tuple[str, ...] = (),
    ) -> None:
        """Set the ``job_states`` answer; the state text defaults to the phase word.

        Args:
            phases: The phase per job or task id.
            states: The raw state text per id, the phase word when ``None``.
            reasons: The one-line reason per id, when the test gives one.
            flags: The flags per id, only ids that have a flag.
            known: Whether the states are real, or the scheduler could not be asked.
            followup_live_job_ids: The ids of the queued or running follow-up jobs.
        """
        raw_states = (
            dict(states)
            if states is not None
            else {task_id: phase.value for task_id, phase in phases.items()}
        )
        self.job_states_answer = JobStates(
            known=known,
            states=MappingProxyType(raw_states),
            phases=MappingProxyType(dict(phases)),
            reasons=MappingProxyType(dict(reasons or {})),
            flags=MappingProxyType(dict(flags or {})),
            followup_live_job_ids=followup_live_job_ids,
        )

    def calls(self, method: str) -> list[Mapping[str, object]]:
        """The keyword arguments of each call of one method, in call order."""
        return [dict(kwargs) for name, kwargs in self._calls if name == method]

    def job_states(
        self,
        *,
        machine: MachineProfile,
        study: Study,
        jobs: Sequence[JobRecord],
        cache: Mapping[str, str],
    ) -> tuple[JobStates, Mapping[str, str]]:
        """The states the test set, with the cache it got, unchanged."""
        self._record(method='job_states', machine=machine, study=study, jobs=jobs, cache=cache)
        return self.job_states_answer, cache

    def expected_starts(self, *, context: Context) -> dict[str, QueuedJob] | None:
        """The starts the test set, as a copy; ``None`` when it set none."""
        self._record(method='expected_starts', context=context)
        if self.expected_starts_answer is None:
            return None
        return dict(self.expected_starts_answer)

    def launch_blocker(self, *, context: Context) -> str | None:
        """A fake launch has no local pid file to block it."""
        return None

    def resources(
        self,
        *,
        machine: MachineProfile,
        overrides: Sequence[MachineOverride],
        grid_name: str,
        values: Mapping[str, str],
        local_cpus: int,
    ) -> Resources:
        """The resources the test set."""
        self._record(
            method='resources',
            machine=machine,
            overrides=overrides,
            grid_name=grid_name,
            values=values,
            local_cpus=local_cpus,
        )
        return self.resources_answer

    def failure_hint(
        self, *, report: RunReport, usage: JobUsage | None, resources: Resources
    ) -> str | None:
        """The hint the test set."""
        self._record(method='failure_hint', report=report, usage=usage, resources=resources)
        return self.failure_hint_answer

    def suggest_sizing(
        self, *, elapsed: timedelta | None, peak_per_cpu: int | None
    ) -> tuple[str | None, str | None]:
        """The suggested time limit and memory per CPU the test set."""
        self._record(method='suggest_sizing', elapsed=elapsed, peak_per_cpu=peak_per_cpu)
        return self.sizing_answer

    def foreground_script(
        self, *, context: Context, resources: Resources, command: Sequence[str]
    ) -> str:
        """A one-line script that runs the command in the foreground."""
        self._record(
            method='foreground_script', context=context, resources=resources, command=command
        )
        return f'exec {shlex.join(command)}\n'

    def launch(  # noqa: PLR0913 # why: the Backend signature
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
        """Submit the groups through the shared launch helper, with a private launcher.

        The launcher hands out ``job-0``, ``job-1``, and so on, counting up
        across every launch of this instance, and writes the same
        ``submissions.jsonl`` records a real launch writes.
        """
        self._record(
            method='launch',
            context=context,
            groups=groups,
            folder=folder,
            dry_run=dry_run,
            start_delay_seconds=start_delay_seconds,
            sleep=sleep,
            task_stream=task_stream,
            environment=environment,
            pid=pid,
            host=host,
            now=now,
        )
        launcher: Launcher = _FakeLauncher(next_job_id=self._take_job_id)
        return launch(
            launcher=launcher,
            write_job_files=_write_no_job_files,
            machine=context.machine,
            study=context.study,
            groups=groups,
            project_root=context.project_root,
            dry_run=dry_run,
            now=now,
            start_delay_seconds=start_delay_seconds,
        )

    def after_submit(
        self,
        *,
        context: Context,
        submitted: Sequence[JobRecord],
        retry: Callable[..., int],
        sleep: Callable[[float], None],
    ) -> int:
        """The exit code the test set."""
        self._record(
            method='after_submit', context=context, submitted=submitted, retry=retry, sleep=sleep
        )
        return self.after_submit_code

    def cancel(
        self, *, context: Context, dry_run: bool, signal: str | None
    ) -> tuple[int, list[str]]:
        """The exit code and ids the test set; the ids are returned as a copy."""
        self._record(method='cancel', context=context, dry_run=dry_run, signal=signal)
        return self.cancel_answer[0], list(self.cancel_answer[1])

    def cancellable_job_ids(self, *, reports: Sequence[RunReport]) -> list[str]:
        """The ids the test set, as a copy."""
        self._record(method='cancellable_job_ids', reports=reports)
        return list(self.cancellable_answer)

    def detach(self, *, context: Context, arguments: Sequence[str], wait_command: str) -> int:
        """The exit code the test set."""
        self._record(
            method='detach', context=context, arguments=arguments, wait_command=wait_command
        )
        return self.detach_code

    def failure_usage(
        self, *, context: Context, reports: Sequence[RunReport]
    ) -> Mapping[str, JobUsage]:
        """The usage the test set, as a read-only mapping."""
        self._record(method='failure_usage', context=context, reports=reports)
        return MappingProxyType(self.failure_usage_answer)

    def usage(self, *, context: Context) -> tuple[Mapping[str, JobUsage], Mapping[str, str | None]]:
        """The usage and GPU ids the test set, as read-only mappings."""
        self._record(method='usage', context=context)
        return MappingProxyType(self.usage_answer), MappingProxyType(self.gpu_ids_answer)

    def doctor_checks(self, *, machine: MachineProfile) -> tuple[DoctorCheck, ...]:
        """The checks the test set."""
        self._record(method='doctor_checks', machine=machine)
        return self.doctor_answer


def install_fake_backend(*, monkeypatch: pytest.MonkeyPatch) -> FakeBackend:
    """Install the fake as the backend of both launchers, and return it.

    Args:
        monkeypatch: The fixture that patches the backend table and reverts it.

    Returns:
        The installed fake, to set the answers of before the command runs.
    """
    fake = FakeBackend()
    monkeypatch.setattr(
        tslab.study.launchers, 'BACKENDS', {LauncherKind.SLURM: fake, LauncherKind.LOCAL: fake}
    )
    return fake


# why: ty checks the fake against the Protocol, so a new Backend member breaks
# the type check until the fake has it
_CHECK: Backend = FakeBackend()

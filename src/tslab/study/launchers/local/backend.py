"""The local backend: the states, resources and launch of a local machine."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TextIO

from tslab.jobs.slurm import SlurmJobUsage
from tslab.study.context import _local_states, Context
from tslab.study.definition import Study
from tslab.study.doctor import DoctorCheck
from tslab.study.launchers import JobPhase, JobStates, LaunchReport, Resources
from tslab.study.machine import MachineOverride, MachineProfile
from tslab.study.plan import _usable_cpus, JobGroup, study_folder
from tslab.study.records import JobRecord
from tslab.study.state import RunReport

__all__ = ['LocalBackend', 'LocalResources']


@dataclass(frozen=True, kw_only=True)
class LocalResources:
    """The resources of a local launch: the CPUs this process may run on."""

    cpus: int
    max_tasks_per_job: int | None = None

    def describe(self) -> str:
        """The resource line of the machine, for the dry run printout."""
        return 'local'


def _phase_of(*, job_state: str | None) -> JobPhase:
    """The phase of one local task state, the SLURM-shaped words of the pid file."""
    if job_state is None:
        return JobPhase.UNKNOWN
    if job_state == 'PENDING':
        return JobPhase.QUEUED
    if job_state == 'RUNNING':
        return JobPhase.RUNNING
    return JobPhase.ENDED


class LocalBackend:
    """The backend of a local machine: in-process tasks under the pid file."""

    #: The launcher of the machines this backend serves.
    name = 'local'

    #: Whether the usage output suggests a memory size.
    suggests_memory = False

    def job_states(
        self,
        *,
        machine: MachineProfile,
        study: Study,
        jobs: Sequence[JobRecord],
        cache: Mapping[str, str],  # why: the local states need no cache
    ) -> tuple[JobStates, Mapping[str, str]]:
        """The states of the tasks; a local machine always knows them."""
        states = _local_states(machine=machine, study=study, jobs=list(jobs))
        phases = {task_id: _phase_of(job_state=state) for task_id, state in states.items()}
        return (
            JobStates(
                known=True,
                states=MappingProxyType(states),
                phases=MappingProxyType(phases),
                followup_live_job_ids=(),
            ),
            cache,
        )

    def expected_starts(
        self,
        *,
        context: Context,  # noqa: ARG002 # why: the Backend signature
    ) -> dict[str, str | None] | None:
        """There is no scheduler to ask; always ``None``."""
        return None

    def resources(
        self,
        *,
        machine: MachineProfile,  # noqa: ARG002 # why: the Backend signature
        overrides: Sequence[MachineOverride],  # noqa: ARG002 # why: the Backend signature
        grid_name: str,  # noqa: ARG002 # why: the Backend signature
        values: Mapping[str, str],  # noqa: ARG002 # why: the Backend signature
    ) -> Resources:
        """The CPUs of the machine; a local launch holds no array limit."""
        return LocalResources(cpus=_usable_cpus())

    def launch(  # noqa: PLR0913 # why: the Backend signature; one keyword per launch input
        self,
        *,
        context: Context,
        groups: Sequence[JobGroup],
        folder: Path,
        dry_run: bool,
        begin: str | None,
        sleep: Callable[[float], None],
        task_stream: TextIO | None,
        environment: Mapping[str, str],  # noqa: ARG002 # why: wired when the launch code moves
    ) -> LaunchReport | None:
        """Run the groups in this process; ``None`` when a problem was logged."""
        # why: the commands module reads the backends door for its types, so a
        # module-level import would cycle
        from tslab.study.commands import _launch_held  # noqa: PLC0415

        return _launch_held(
            context=context,
            groups=groups,
            folder=folder,
            dry_run=dry_run,
            begin=begin,
            sleep=sleep,
            task_stream=task_stream,
        )

    def after_submit(
        self,
        *,
        context: Context,
        submitted: Sequence[JobRecord],
        retry: Callable[..., int],
        sleep: Callable[[float], None],  # noqa: ARG002 # why: the Backend signature
    ) -> int:
        """The in-process auto-retry of the interrupted runs; 0 without it."""
        if context.machine.followup.auto_retry and submitted:
            return retry()
        return 0

    def cancel(
        self, *, context: Context, dry_run: bool, signal: str | None
    ) -> tuple[int, list[str]]:
        """Stop the live local launch; the exit code and the stopped pid."""
        # why: the commands module reads the backends door for its types, so a
        # module-level import would cycle
        from tslab.study.commands import cancel  # noqa: PLC0415

        return cancel(context=context, dry_run=dry_run, signal=signal)

    def cancellable_job_ids(
        self,
        *,
        reports: Sequence[RunReport],  # noqa: ARG002 # why: the Backend signature
    ) -> list[str]:
        """A local launch is one unit; there is no single job id to cancel."""
        return []

    def detach(self, *, context: Context, arguments: Sequence[str], wait_command: str) -> int:
        """Start the launch in the background and wait until it holds the pid file."""
        # why: local_launch reads the backends door for OnTaskState, so a
        # module-level import would cycle
        from tslab.study.local_launch import detach_launch  # noqa: PLC0415

        return detach_launch(
            study_folder=study_folder(machine=context.machine, study=context.study),
            arguments=arguments,
            project_root=context.project_root,
            wait_command=wait_command,
        )

    def failure_usage(
        self,
        *,
        context: Context,  # noqa: ARG002 # why: the Backend signature
        reports: Sequence[RunReport],  # noqa: ARG002 # why: the Backend signature
    ) -> Mapping[str, SlurmJobUsage]:
        """There is no accounting on a local machine."""
        no_usage: dict[str, SlurmJobUsage] = {}
        return no_usage

    def usage(
        self, *, context: Context
    ) -> tuple[Mapping[str, SlurmJobUsage], Mapping[str, str | None]]:
        """The usage of every local run from its usage file, with the GPU ids."""
        # why: the usage module reads the launch module, which reads the backends
        # door, so a module-level import would cycle
        from tslab.study.usage import _local_usage_of  # noqa: PLC0415

        return _local_usage_of(context=context)

    def doctor_checks(self) -> tuple[DoctorCheck, ...]:
        """A local machine needs no scheduler checks."""
        return ()

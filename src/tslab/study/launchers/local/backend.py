"""The local backend: the states, resources and launch of a local machine."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
import functools
import logging
import os
from pathlib import Path
from types import MappingProxyType
from typing import TextIO

from tslab.core.meta import hostname
from tslab.jobs.local import start_keep_awake, stop_keep_awake, visible_gpus
from tslab.study.context import Context
from tslab.study.definition import Study
from tslab.study.doctor import DoctorCheck
from tslab.study.launchers import (
    JobPhase,
    JobStates,
    JobUsage,
    launch,
    LaunchReport,
    OnTaskState,
    QueuedJob,
    Resources,
)
from tslab.study.launchers.local.foreground import foreground_script
from tslab.study.launchers.local.pid_file import _check_free, hold_launch, LocalLaunchError
from tslab.study.launchers.local.run import (
    local_job_id,
    LocalLauncher,
    resolve_local_gpus,
    write_job_files,
)
from tslab.study.launchers.local.states import _local_states
from tslab.study.machine import LocalSettings, MachineOverride, MachineProfile
from tslab.study.paths import study_folder
from tslab.study.plan import JobGroup
from tslab.study.records import JobRecord
from tslab.study.reporting.state import RunReport

__all__ = ['LocalBackend', 'LocalResources']

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class LocalResources:
    """The resources of a local launch: the CPUs this process may run on.

    Attributes:
        cpus: The CPUs this process may run on.
        cpus_per_run: The CPUs of one run when several run at once.
        max_tasks_per_job: A local launch holds no array limit.
    """

    cpus: int
    cpus_per_run: int
    max_tasks_per_job: int | None = None

    @property
    def requested_memory(self) -> int | None:
        """A local launch requests no memory."""
        return None

    def describe(self) -> str:
        """The resource line of the machine, for the dry run printout."""
        return 'local'

    def as_record(self) -> dict[str, object]:
        """A local launch records no resource fields."""
        return {}

    def signal_problem(self) -> str | None:
        """A local profile checks time-limit signals when it loads.

        Returns:
            Always ``None``.
        """
        return None


@functools.cache
def _auto_gpu_count() -> int:
    """The number of visible GPUs, looked up once per process."""
    return len(visible_gpus(environment=os.environ))


def _phase_of(*, job_state: str | None) -> JobPhase:
    """The phase of one local task state from the pid file."""
    if job_state is None:
        return JobPhase.UNKNOWN
    if job_state == 'pending':
        return JobPhase.QUEUED
    if job_state == 'running':
        return JobPhase.RUNNING
    return JobPhase.ENDED


class LocalBackend:
    """The backend of a local machine: in-process tasks under the pid file."""

    #: The launcher of the machines this backend serves.
    name = 'local'

    #: Whether the jobs of this backend have a time limit.
    has_time_limit = False
    knows_other_hosts = False

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
                reasons=MappingProxyType({}),
                flags=MappingProxyType({}),
                followup_live_job_ids=(),
            ),
            cache,
        )

    def expected_starts(
        self,
        *,
        context: Context,  # noqa: ARG002 # why: the Backend signature
    ) -> dict[str, QueuedJob] | None:
        """There is no scheduler to ask; always ``None``."""
        return None

    def launch_blocker(self, *, context: Context) -> str | None:
        """Why a new local launch of this study cannot start now, or ``None``."""
        try:
            _check_free(
                study_folder=study_folder(machine=context.machine, study=context.study),
                host=hostname(),
            )
        except LocalLaunchError as error:
            return str(error)
        return None

    def resources(
        self,
        *,
        machine: MachineProfile,
        overrides: Sequence[MachineOverride],  # noqa: ARG002 # why: the Backend signature
        grid_name: str,  # noqa: ARG002 # why: the Backend signature
        values: Mapping[str, str],  # noqa: ARG002 # why: the Backend signature
        local_cpus: int,
    ) -> Resources:
        """The CPUs of the machine; a local launch holds no array limit."""
        local = machine.local or LocalSettings()
        # ponytail: skip_busy_gpus is not applied here: fewer slots at launch only means each
        # run got a smaller share than it could have (safe direction)
        if local.gpus == 'auto':
            slots = _auto_gpu_count() * local.tasks_per_gpu or local.max_parallel
        elif local.gpus:
            slots = len(local.gpus) * local.tasks_per_gpu
        else:
            slots = local.max_parallel
        return LocalResources(cpus=local_cpus, cpus_per_run=max(local_cpus // slots, 1))

    def failure_hint(
        self,
        *,
        report: RunReport,  # noqa: ARG002 # why: the Backend signature
        usage: JobUsage | None,  # noqa: ARG002 # why: the Backend signature
        resources: Resources,  # noqa: ARG002 # why: the Backend signature
    ) -> str | None:
        """A local machine has no scheduler state to hint about."""
        return None

    def suggest_sizing(
        self,
        *,
        elapsed: timedelta | None,  # noqa: ARG002 # why: the Backend signature
        peak_per_cpu: int | None,  # noqa: ARG002 # why: the Backend signature
    ) -> tuple[str | None, str | None]:
        """A local machine has no time limit or memory request to size."""
        return None, None

    def foreground_script(
        self,
        *,
        context: Context,
        resources: Resources,  # noqa: ARG002 # why: the Backend signature
        command: Sequence[str],
    ) -> str:
        """The foreground script of the machine: no setup lines."""
        return foreground_script(
            project_root=context.project_root, env=context.machine.env, command=command
        )

    def launch(  # noqa: PLR0913 # why: the Backend signature; one keyword per launch input
        self,
        *,
        context: Context,
        groups: Sequence[JobGroup],
        folder: Path,
        dry_run: bool,
        start_delay_seconds: int | None,
        sleep: Callable[[float], None],  # noqa: ARG002 # why: the Backend launch signature
        task_stream: TextIO | None,
        environment: Mapping[str, str],
        pid: int,
        host: str,
        now: Callable[[], str],
    ) -> LaunchReport | None:
        """Run the groups in this process; ``None`` when a problem was logged."""
        machine = context.machine
        local = machine.local or LocalSettings()
        gpu_ids: tuple[str, ...] = ()
        keep_awake = None
        if not dry_run:
            try:
                gpu_ids = resolve_local_gpus(local=local, environment=environment)
            except ValueError as error:
                _LOGGER.error('%s', error)  # noqa: TRY400 # why: a GPU setup error is shown, not raised
                return None
            _LOGGER.info('local GPUs: %s', ','.join(gpu_ids) if gpu_ids else 'none')
        if not dry_run and local.keep_awake:
            keep_awake = start_keep_awake(pid=pid, reason='tslab study launch')
        if dry_run:
            return _launch_local_groups(
                context=context,
                groups=groups,
                dry_run=True,
                start_delay_seconds=start_delay_seconds,
                task_stream=task_stream,
                local=local,
                gpu_ids=gpu_ids,
                on_task_state=None,
                pid=pid,
                host=host,
                now=now,
            )
        try:
            task_ids = [
                f'{local_job_id(job_file=group.job_file, pid=pid)}_{task.index}'
                for group in groups
                for task in group.tasks
            ]
            with hold_launch(
                study_folder=folder, task_ids=task_ids, pid=pid, host=host, now=now()
            ) as on_task_state:
                return _launch_local_groups(
                    context=context,
                    groups=groups,
                    dry_run=False,
                    start_delay_seconds=start_delay_seconds,
                    task_stream=task_stream,
                    local=local,
                    gpu_ids=gpu_ids,
                    on_task_state=on_task_state,
                    pid=pid,
                    host=host,
                    now=now,
                )
        except LocalLaunchError as error:
            _LOGGER.error('%s', error)  # noqa: TRY400 # why: a launch-file problem is shown, not raised
            return None
        finally:
            stop_keep_awake(process=keep_awake)

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
        from tslab.study.launchers.local.cancel import cancel_local  # noqa: PLC0415

        return cancel_local(context=context, dry_run=dry_run, signal=signal)

    def cancellable_job_ids(
        self,
        *,
        reports: Sequence[RunReport],  # noqa: ARG002 # why: the Backend signature
    ) -> list[str]:
        """A local launch is one unit; there is no single job id to cancel."""
        return []

    def detach(self, *, context: Context, arguments: Sequence[str], wait_command: str) -> int:
        """Start the launch in the background and wait until it holds the pid file."""
        # why: the local pid-file module reads the backends door for OnTaskState, so a
        # module-level import would cycle
        from tslab.study.launchers.local.pid_file import detach_launch  # noqa: PLC0415

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
    ) -> Mapping[str, JobUsage]:
        """There is no accounting on a local machine."""
        no_usage: dict[str, JobUsage] = {}
        return no_usage

    def usage(self, *, context: Context) -> tuple[Mapping[str, JobUsage], Mapping[str, str | None]]:
        """The usage of every local run from its usage file, with the GPU ids."""
        from tslab.study.launchers.local.usage import usage  # noqa: PLC0415

        return usage(context=context)

    def doctor_checks(self) -> tuple[DoctorCheck, ...]:
        """A local machine needs no scheduler checks."""
        return ()


def _launch_local_groups(  # noqa: PLR0913 # why: the shared launch adapter forwards backend inputs
    *,
    context: Context,
    groups: Sequence[JobGroup],
    dry_run: bool,
    start_delay_seconds: int | None,
    task_stream: TextIO | None,
    local: LocalSettings,
    gpu_ids: tuple[str, ...],
    on_task_state: OnTaskState | None,
    pid: int,
    host: str,
    now: Callable[[], str],
) -> LaunchReport:
    return launch(
        launcher=LocalLauncher(
            project_root=context.project_root,
            env=context.machine.env,
            local=local,
            gpu_ids=gpu_ids,
            pid=pid,
            host=host,
            now=now,
            stream=task_stream,
            on_task_state=on_task_state,
        ),
        write_job_files=write_job_files,
        machine=context.machine,
        study=context.study,
        groups=groups,
        project_root=context.project_root,
        dry_run=dry_run,
        start_delay_seconds=start_delay_seconds,
        now=now,
    )

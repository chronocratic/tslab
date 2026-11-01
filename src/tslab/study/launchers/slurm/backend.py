"""The SLURM backend: the states, resources and launch of a SLURM machine."""

from collections.abc import Callable, Mapping, Sequence
import logging
from pathlib import Path
from types import MappingProxyType
from typing import TextIO

from tslab.jobs.slurm import (
    FAILING_JOB_STATES,
    INTERRUPTING_JOB_STATES,
    QUEUED_JOB_STATES,
    RUNNING_JOB_STATES,
    slurm_available,
    SlurmError,
    SlurmJobUsage,
)
from tslab.study.context import _query_states, Context, expected_starts
from tslab.study.definition import Study
from tslab.study.doctor import _slurm_check, DoctorCheck
from tslab.study.launchers import JobPhase, JobStates, LaunchReport, Resources
from tslab.study.machine import MachineOverride, MachineProfile, MachineProfileError
from tslab.study.plan import _slurm_for, JobGroup
from tslab.study.records import JobRecord
from tslab.study.state import followup_job_ids, RunReport, slurm_task_ids

__all__ = ['SlurmBackend']

_LOGGER = logging.getLogger(__name__)

#: The SLURM state sets that give one phase, in the order the logic checks them.
_PHASE_TABLE: tuple[tuple[frozenset[str], JobPhase], ...] = (
    (frozenset(QUEUED_JOB_STATES), JobPhase.QUEUED),
    (frozenset(RUNNING_JOB_STATES), JobPhase.RUNNING),
    (frozenset(INTERRUPTING_JOB_STATES), JobPhase.INTERRUPTED),
    (frozenset(FAILING_JOB_STATES), JobPhase.FAILED),
    (frozenset(('COMPLETED',)), JobPhase.COMPLETED),
)


def _phase_of(*, job_state: str | None) -> JobPhase:
    """The phase of one SLURM task state; an unknown state is over."""
    if job_state is None:
        return JobPhase.UNKNOWN
    for states, phase in _PHASE_TABLE:
        if job_state in states:
            return phase
    return JobPhase.ENDED


def _unknown_states() -> JobStates:
    """The states when the scheduler cannot be asked."""
    return JobStates(
        known=False,
        states=MappingProxyType({}),
        phases=MappingProxyType({}),
        followup_live_job_ids=(),
    )


class SlurmBackend:
    """The backend of a SLURM machine: sbatch jobs, sacct states, accounting."""

    #: The launcher of the machines this backend serves.
    name = 'slurm'

    #: Whether the usage output suggests a memory size.
    suggests_memory = True

    def job_states(
        self,
        *,
        machine: MachineProfile,  # noqa: ARG002 # why: the Backend signature
        study: Study,  # noqa: ARG002 # why: the Backend signature
        jobs: Sequence[JobRecord],
        cache: Mapping[str, str],
    ) -> tuple[JobStates, Mapping[str, str]]:
        """The states of the tasks and follow-up jobs, with the new cache second."""
        if not slurm_available():
            return _unknown_states(), cache
        ended_states = dict(cache)
        try:
            followups = followup_job_ids(jobs=jobs)
            states = _query_states(
                job_ids=slurm_task_ids(jobs=jobs) + followups, ended_states=ended_states
            )
        except SlurmError as error:
            _LOGGER.warning('%s', error)
            return _unknown_states(), cache
        live = tuple(
            job_id
            for job_id in followups
            if states.get(job_id) in QUEUED_JOB_STATES | RUNNING_JOB_STATES
        )
        phases = {task_id: _phase_of(job_state=state) for task_id, state in states.items()}
        return (
            JobStates(
                known=True,
                states=MappingProxyType(states),
                phases=MappingProxyType(phases),
                followup_live_job_ids=live,
            ),
            ended_states,
        )

    def expected_starts(self, *, context: Context) -> dict[str, str | None] | None:
        """The expected start of the queued jobs; ``None`` when nothing is queued."""
        return expected_starts(context=context)

    def resources(
        self,
        *,
        machine: MachineProfile,
        overrides: Sequence[MachineOverride],
        grid_name: str,
        values: Mapping[str, str],
    ) -> Resources:
        """The resources of one grid on the machine, with its overrides applied."""
        settings = _slurm_for(
            base=machine.slurm, overrides=overrides, grid_name=grid_name, values=values
        )
        if settings is None:
            message = f'{machine.path}: a slurm launcher needs the [slurm] table'
            raise MachineProfileError(message)
        return settings

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
        """Submit the groups with sbatch; ``None`` when a problem was logged."""
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
        retry: Callable[..., int],  # noqa: ARG002 # why: the Backend signature
        sleep: Callable[[float], None],
    ) -> int:
        """The exit code of the queued follow-up jobs; 0 without one."""
        # why: the followups module reads the launch module, which reads the backends
        # door, so a module-level import would cycle
        from tslab.study.followups import queue_followups  # noqa: PLC0415

        code, _ = queue_followups(
            context=context, job_ids=[record.job_id for record in submitted], sleep=sleep
        )
        return code

    def cancel(
        self, *, context: Context, dry_run: bool, signal: str | None
    ) -> tuple[int, list[str]]:
        """Cancel or signal the queued and running jobs; the exit code and ids."""
        # why: the commands module reads the backends door for its types, so a
        # module-level import would cycle
        from tslab.study.commands import cancel  # noqa: PLC0415

        return cancel(context=context, dry_run=dry_run, signal=signal)

    def cancellable_job_ids(self, *, reports: Sequence[RunReport]) -> list[str]:
        """The job ids a cancel can reach: the queued and running runs."""
        # why: the commands module reads the backends door for its types, so a
        # module-level import would cycle
        from tslab.study.commands import LIVE_STATES  # noqa: PLC0415

        return [
            report.job_id
            for report in reports
            if report.state in LIVE_STATES and report.job_id is not None
        ]

    def detach(
        self,
        *,
        context: Context,  # noqa: ARG002 # why: the Backend signature
        arguments: Sequence[str],  # noqa: ARG002 # why: the Backend signature
        wait_command: str,  # noqa: ARG002 # why: the Backend signature
    ) -> int:
        """SLURM jobs already run in the background; detaching is a usage error.

        Raises:
            ValueError: Always; the CLI turns it into a ``UsageError``.
        """
        message = '--detach is for local machines; SLURM jobs already run in the background'
        raise ValueError(message)

    def failure_usage(
        self, *, context: Context, reports: Sequence[RunReport]
    ) -> Mapping[str, SlurmJobUsage]:
        """The accounting usage of the out-of-memory runs; no numbers on a failure."""
        # why: the failures module reads the report module, which reads the launch
        # module and the backends door, so a module-level import would cycle
        from tslab.study.failures import _usage_of  # noqa: PLC0415

        return _usage_of(context=context, reports=reports)

    def usage(
        self, *, context: Context
    ) -> tuple[Mapping[str, SlurmJobUsage], Mapping[str, str | None]]:
        """The accounting usage of every submitted run; no GPU ids on SLURM."""
        # why: the usage module reads the launch module, which reads the backends
        # door, so a module-level import would cycle
        from tslab.study.usage import _usage_of  # noqa: PLC0415

        no_gpus: dict[str, str | None] = {}
        return _usage_of(context=context), no_gpus

    def doctor_checks(self) -> tuple[DoctorCheck, ...]:
        """The readiness checks of a SLURM machine."""
        return (_slurm_check(),)

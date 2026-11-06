"""The SLURM backend: the states, resources and launch of a SLURM machine."""

from collections.abc import Callable, Mapping, Sequence
from datetime import timedelta
from functools import partial
import logging
from pathlib import Path
from types import MappingProxyType
from typing import cast, TextIO

from tslab.core.status import utc_now_iso
from tslab.enums import CheckStatus, JobFlag, JobPhase, LauncherKind, StudyState
from tslab.jobs.slurm import (
    FAILING_JOB_STATES,
    format_slurm_duration,
    format_slurm_memory,
    INTERRUPTING_JOB_STATES,
    query_job_reasons,
    QUEUED_JOB_STATES,
    RUNNING_JOB_STATES,
    scancel,
    slurm_available,
    SlurmError,
)
from tslab.study.context import Context
from tslab.study.definition import chained_grids, Study
from tslab.study.doctor import DoctorCheck
from tslab.study.launchers.common.launch import launch
from tslab.study.launchers.common.types import (
    JobStates,
    JobUsage,
    LaunchError,
    LaunchReport,
    QueuedJob,
    Resources,
)
from tslab.study.launchers.slurm.doctor import cluster_checks, slurm_check
from tslab.study.launchers.slurm.hints import state_hint
from tslab.study.launchers.slurm.job_script import foreground_script
from tslab.study.launchers.slurm.records import SlurmJobRecord
from tslab.study.launchers.slurm.slurm_file import write_slurm_file
from tslab.study.launchers.slurm.states import (
    _query_states,
    expected_starts,
    followup_job_ids,
    slurm_task_ids,
)
from tslab.study.launchers.slurm.submit import SlurmLauncher, write_job_files
from tslab.study.machine import MachineOverride, MachineProfile, MachineProfileError, SlurmSettings
from tslab.study.machine.slurm import slurm_for
from tslab.study.paths import study_folder
from tslab.study.plan import JobGroup
from tslab.study.records import JobRecord, submission_lock
from tslab.study.reporting.state import RunReport

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


#: What the SLURM states mean for the state logic; other states carry no flag.
_FLAGS: Mapping[str, frozenset[JobFlag]] = MappingProxyType(
    {
        'TIMEOUT': frozenset({JobFlag.TIME_LIMIT}),
        'PREEMPTED': frozenset({JobFlag.NOT_COUNTED}),
        'NODE_FAIL': frozenset({JobFlag.NOT_COUNTED}),
        'CANCELLED': frozenset({JobFlag.CANCELLED}),
    }
)

_DEPENDENCY_REASON_FLAGS: Mapping[str, frozenset[JobFlag]] = MappingProxyType(
    {
        'Dependency': frozenset({JobFlag.DEPENDENCY_PENDING}),
        'DependencyNeverSatisfied': frozenset({JobFlag.DEPENDENCY_NEVER}),
    }
)


def _flags_of(
    *, states: Mapping[str, str], reasons: Mapping[str, str]
) -> dict[str, frozenset[JobFlag]]:
    """Combine state and dependency-reason flags for each task."""
    flags = {
        task_id: flags
        for task_id, state in states.items()
        if (flags := _FLAGS.get(state)) is not None
    }
    for task_id, reason in reasons.items():
        reason_flags = _DEPENDENCY_REASON_FLAGS.get(reason)
        if reason_flags is None:
            continue
        flags[task_id] = flags.get(task_id, frozenset()) | reason_flags
    return flags


def _unknown_states() -> JobStates:
    """The states when the scheduler cannot be asked."""
    return JobStates(
        known=False,
        states=MappingProxyType({}),
        phases=MappingProxyType({}),
        reasons=MappingProxyType({}),
        flags=MappingProxyType({}),
        followup_live_job_ids=(),
    )


class SlurmBackend:
    """The backend of a SLURM machine: sbatch jobs, sacct states, accounting."""

    #: The launcher of the machines this backend serves.
    name = LauncherKind.SLURM

    #: Whether the jobs of this backend have a time limit.
    has_time_limit = True
    knows_other_hosts = True

    def job_states(
        self,
        *,
        machine: MachineProfile,  # noqa: ARG002 # why: the Backend signature
        study: Study,
        jobs: Sequence[JobRecord],
        cache: Mapping[str, str],
    ) -> tuple[JobStates, Mapping[str, str]]:
        """The states of the tasks and follow-up jobs, with the new cache second."""
        if not slurm_available():
            return _unknown_states(), cache
        try:
            followups = followup_job_ids(jobs=jobs)
            task_ids = slurm_task_ids(jobs=jobs)
            states, new_cache = _query_states(job_ids=task_ids + followups, ended_states=cache)
            reasons = query_job_reasons(job_ids=task_ids) if chained_grids(study=study) else {}
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
                reasons=MappingProxyType(
                    {
                        task_id: f'SLURM {state}'
                        for task_id, state in states.items()
                        if state.isupper()
                    }
                ),
                flags=MappingProxyType(_flags_of(states=states, reasons=reasons)),
                followup_live_job_ids=live,
            ),
            new_cache,
        )

    def expected_starts(self, *, context: Context) -> dict[str, QueuedJob] | None:
        """The expected start and pending reason of the queued jobs.

        ``None`` when nothing is queued.
        """
        return expected_starts(context=context)

    def launch_blocker(self, *, context: Context) -> str | None:  # noqa: ARG002 # why: Backend API
        """A SLURM launch has no local pid file to block it."""
        return None

    def resources(
        self,
        *,
        machine: MachineProfile,
        overrides: Sequence[MachineOverride],
        grid_name: str,
        values: Mapping[str, str],
        local_cpus: int,  # noqa: ARG002 # why: the Backend signature
    ) -> Resources:
        """The resources of one grid on the machine, with its overrides applied."""
        settings = slurm_for(
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
        folder: Path,  # noqa: ARG002 # why: the Backend launch signature
        dry_run: bool,
        start_delay_seconds: int | None,
        sleep: Callable[[float], None],
        task_stream: TextIO | None,  # noqa: ARG002 # why: the Backend launch signature
        environment: Mapping[str, str],  # noqa: ARG002 # why: the SLURM writer uses the profile env
        pid: int,  # noqa: ARG002 # why: the Backend launch signature
        host: str,  # noqa: ARG002 # why: the Backend launch signature
        now: Callable[[], str],
    ) -> LaunchReport | None:
        """Submit the groups with sbatch; ``None`` when a problem was logged."""
        chain_check = _chain_check(context=context, groups=groups)
        if any(group.chained for group in groups) and chain_check is None:
            return None
        return launch(
            launcher=SlurmLauncher(sleep=sleep),
            record_type=SlurmJobRecord,
            write_job_files=partial(write_job_files, chain_check=chain_check),
            machine=context.machine,
            study=context.study,
            groups=groups,
            project_root=context.project_root,
            dry_run=dry_run,
            start_delay_seconds=start_delay_seconds,
            now=now,
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
        from tslab.study.launchers.slurm.followups import queue_followups  # noqa: PLC0415

        code, _ = queue_followups(
            context=context,
            job_ids=[record.job_id for record in submitted],
            now=utc_now_iso,
            sleep=sleep,
        )
        if submitted:
            self._write_slurm_file(context=context)
        return code

    def after_stopped_submit(self, *, context: Context, submitted: Sequence[JobRecord]) -> None:
        """Write ``slurm.sh`` for the jobs queued before the stop."""
        if submitted:
            self._write_slurm_file(context=context)

    @staticmethod
    def _write_slurm_file(*, context: Context) -> None:
        """Write ``slurm.sh``; a failure only warns, the jobs are already queued."""
        folder = study_folder(machine=context.machine, study=context.study)
        try:
            with submission_lock(study_folder=folder):
                path = write_slurm_file(study=context.study, study_folder=folder)
        except OSError as error:
            # why: the jobs are already queued, so the exit code stays as it is
            _LOGGER.warning('could not write the SLURM commands file: %s', error)
            return
        _LOGGER.info('SLURM commands: source %s && squeue-full-name', path)

    def cancel(
        self, *, context: Context, dry_run: bool, signal: str | None
    ) -> tuple[int, list[str]]:
        """Cancel or signal the queued and running jobs; the exit code and ids."""
        live = _live_reports(reports=context.reports, signal=signal)
        job_ids = [report.job_id for report in live if report.job_id is not None]
        if not job_ids:
            _LOGGER.info('nothing to cancel')
            return 0, []
        if dry_run:
            return _dry_run_cancel(job_ids=job_ids, signal=signal)
        try:
            scancel(job_ids=job_ids, signal=signal)
        except SlurmError as error:
            raise LaunchError(str(error)) from error
        _log_cancelled(job_ids=job_ids, signal=signal)
        return 0, job_ids

    def cancellable_job_ids(self, *, reports: Sequence[RunReport]) -> list[str]:
        """The job ids a cancel can reach: the queued and running runs."""
        return [
            report.job_id
            for report in reports
            if report.state in (StudyState.QUEUED, StudyState.RUNNING) and report.job_id is not None
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

    def failure_hint(
        self, *, report: RunReport, usage: JobUsage | None, resources: Resources
    ) -> str | None:
        """The hint of the final SLURM state of the run."""
        return state_hint(job_state=report.job_state, usage=usage, resources=resources)

    def suggest_sizing(
        self, *, elapsed: timedelta | None, peak_per_cpu: int | None
    ) -> tuple[str | None, str | None]:
        """The sbatch ``--time`` and ``--mem-per-cpu`` texts of the sizes."""
        return (
            None if elapsed is None else format_slurm_duration(duration=elapsed),
            None if peak_per_cpu is None else format_slurm_memory(size=peak_per_cpu),
        )

    def foreground_script(
        self, *, context: Context, resources: Resources, command: Sequence[str]
    ) -> str:
        """The foreground script with the setup lines of the SLURM settings."""
        settings = cast('SlurmSettings', resources)
        if settings.gpus > 1:
            _LOGGER.info('study run starts one process; the job would start %d', settings.gpus)
        return foreground_script(
            slurm=settings,
            project_root=context.project_root,
            env=context.machine.env,
            command=command,
        )

    def failure_usage(
        self, *, context: Context, reports: Sequence[RunReport]
    ) -> Mapping[str, JobUsage]:
        """The accounting usage of the failed and interrupted runs; no numbers on a failure."""
        from tslab.study.launchers.slurm.usage import failure_usage  # noqa: PLC0415

        return failure_usage(context=context, reports=reports)

    def usage(self, *, context: Context) -> tuple[Mapping[str, JobUsage], Mapping[str, str | None]]:
        """The accounting usage of every submitted run; no GPU ids on SLURM.

        Raises:
            LaunchError: If the accounting query fails; the message is the
                scheduler error.
        """
        from tslab.study.launchers.slurm.usage import usage  # noqa: PLC0415

        no_gpus: dict[str, str | None] = {}
        try:
            return usage(context=context), no_gpus
        except SlurmError as error:
            raise LaunchError(str(error)) from error

    def doctor_checks(self, *, machine: MachineProfile) -> tuple[DoctorCheck, ...]:
        """The readiness checks of a SLURM machine, against the cluster when it answers.

        The cluster checks are skipped when the command check fails: there is
        no scheduler to ask the cluster to.
        """
        check = slurm_check()
        if check.status != CheckStatus.OK:
            return (check,)
        return (check, *cluster_checks(slurm=machine.slurm))


def _chain_check(*, context: Context, groups: Sequence[JobGroup]) -> tuple[str, ...] | None:
    """The command prefix used to check each chained SLURM task after it runs."""
    if not any(group.chained for group in groups):
        return None
    if not context.reference:
        message = 'a study reference is required to submit chained SLURM jobs'
        _LOGGER.error('%s', message)
        return None
    return (
        *context.machine.tslab_command,
        'study',
        'chain-exit',
        context.reference,
        '--machine',
        context.machine_path.as_posix(),
    )


def _live_reports(*, reports: Sequence[RunReport], signal: str | None) -> list[RunReport]:
    live = [
        report
        for report in reports
        if report.state in (StudyState.QUEUED, StudyState.RUNNING) and report.job_id is not None
    ]
    if signal is None:
        return live
    queued = [
        report.job_id
        for report in live
        if report.state is StudyState.QUEUED and report.job_id is not None
    ]
    _log_skipped_queued(job_ids=queued)
    return [report for report in live if report.state is StudyState.RUNNING]


def _log_skipped_queued(*, job_ids: Sequence[str]) -> None:
    if job_ids:
        _LOGGER.info('skipped queued runs: %s', ', '.join(job_ids))


def _dry_run_cancel(*, job_ids: Sequence[str], signal: str | None) -> tuple[int, list[str]]:
    if signal is None:
        _LOGGER.info('would cancel %s', ', '.join(job_ids))
        return 0, list(job_ids)
    _LOGGER.info(
        'would signal %s with %s; the jobs save and end on their own', ', '.join(job_ids), signal
    )
    return 0, list(job_ids)


def _log_cancelled(*, job_ids: Sequence[str], signal: str | None) -> None:
    if signal is None:
        _LOGGER.info('cancelled %s', ', '.join(job_ids))
        return
    _LOGGER.info(
        'signalled %s with %s; the jobs save and end on their own', ', '.join(job_ids), signal
    )

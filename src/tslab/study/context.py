"""What a study command reads before it does anything: the ``Context``."""

import dataclasses
from dataclasses import dataclass
import logging
from pathlib import Path

from tslab.jobs.slurm import query_expected_starts, query_job_states, slurm_available, SlurmError
from tslab.study.definition import expand_runs, select_runs, Study, StudyRun
from tslab.study.loading import load_study
from tslab.study.local_launch import local_job_states
from tslab.study.machine import load_machine, MachineProfile
from tslab.study.plan import check_machine, resolve_runs, ResolvedRun, study_folder
from tslab.study.records import JobRecord, latest_tasks, read_jobs, submission_in_progress
from tslab.study.state import (
    collect_states,
    followup_job_ids,
    job_ended,
    LOCAL_ENDED,
    QUEUED_JOB_STATES,
    RUNNING_JOB_STATES,
    RunReport,
    slurm_task_ids,
    StudyState,
)

__all__ = [
    'Context',
    'Selection',
    'context_for',
    'expected_starts',
    'load_context',
    'refresh_context',
]

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class Selection:
    """Which runs of a study a command takes.

    Args:
        grids: The grid names to keep; all of them when empty.
        only: ``axis=value`` matches, comma separated per flag.
    """

    grids: tuple[str, ...] = ()
    only: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class Context:
    """What one study command reads before it does anything.

    Args:
        study: The study the command runs on.
        reference: The ``module:attribute`` reference the study came from.
        machine: The machine the runs run on.
        machine_path: The profile file, for messages and follow-up jobs.
        project_root: The folder the jobs run in, the one at command time.
        runs: The selected runs, in expansion order.
        resolved: The selected runs with the machine applied, by run path.
        jobs: The job records of the study, in submission order.
        reports: The state of every selected run.
        job_states_known: Whether the SLURM states the reports use are real.
        submission_in_progress: Whether another process is submitting runs.
        followup_live: Whether a SLURM follow-up job is queued or running.
        followup_live_job_ids: The ids of queued or running SLURM follow-up jobs.
    """

    study: Study
    reference: str
    machine: MachineProfile
    machine_path: Path
    project_root: Path
    runs: list[StudyRun]
    resolved: dict[str, ResolvedRun]
    jobs: list[JobRecord]
    reports: list[RunReport]
    job_states_known: bool
    submission_in_progress: bool = False
    followup_live: bool = False
    followup_live_job_ids: tuple[str, ...] = ()

    @property
    def study_directory(self) -> Path:
        """The ``<output_root>/<study>`` folder that holds the runs."""
        return self.machine.output_root / self.study.name


def expected_starts(*, context: Context) -> dict[str, str | None] | None:
    """The expected start times of the queued runs, keyed by job id.

    Args:
        context: The context to read the queued runs from.

    Returns:
        The expected start of every queued run, keyed by its SLURM job id;
        ``None`` on a local machine, when no run is queued, or when the
        scheduler call fails (a failed call is logged as a warning).
    """
    if context.machine.launcher != 'slurm':
        return None
    queued = [
        report.job_id
        for report in context.reports
        if report.state is StudyState.QUEUED and report.job_id is not None
    ]
    if not queued:
        return None
    try:
        return query_expected_starts(job_ids=queued)
    except SlurmError as error:
        _LOGGER.warning('%s', error)
        return None


def context_for(  # noqa: PLR0913 # why: keyword-only public API; each option is documented below
    *,
    study: Study,
    machine_path: Path,
    selection: Selection,
    project_root: Path,
    reference: str = '',
    read_only: bool = False,
    ended_states: dict[str, str] | None = None,
) -> Context:
    """The machine, runs, job records and states of one study.

    The settings of the machine are checked against the study here, so every
    command reports a bad profile, not only the ones that submit.

    Args:
        study: The study the context is for.
        machine_path: The machine profile file to load.
        selection: The grids and ``only`` matches to keep.
        project_root: The folder the command was called from; the jobs
            ``cd`` into it.
        reference: The study reference, for messages and follow-up jobs;
            ``''`` when the study was not loaded from one, which turns the
            automatic retry follow-up off.
        read_only: For commands that never launch a job: an unset ``${VAR}``
            in the ``[env]`` of the profile is tolerated.
        ended_states: A cache of final SLURM states by job id, kept by a caller that
            loads the context again and again. Ids in it are not queried; ids that
            ended in this query are added to it. ``None`` queries every id.

    Returns:
        The context of the study.

    Raises:
        MachineProfileError: If the machine file does not hold a profile.
        ValueError: If a setting path of the profile is not a field of a grid.
    """
    machine = load_machine(path=machine_path, lenient_env=read_only)
    # One set per command, so a dead override warns once, not once per check.
    warned: set[str] = set()
    check_machine(study=study, machine=machine, warned=warned)
    runs = select_runs(
        study=study, runs=expand_runs(study=study), grids=selection.grids, only=selection.only
    )
    resolved = resolve_runs(study=study, machine=machine, runs=runs, warned=warned)
    jobs = read_jobs(path=study_folder(machine=machine, study=study) / 'submissions.jsonl')
    job_states: dict[str, str] | None = None
    job_states_known = True
    followup_live_job_ids: tuple[str, ...] = ()
    if machine.launcher == 'slurm':
        job_states_known = slurm_available()
        if job_states_known:
            try:
                followups = followup_job_ids(jobs=jobs)
                job_states = _query_states(
                    job_ids=slurm_task_ids(jobs=jobs) + followups, ended_states=ended_states
                )
                followup_live_job_ids = tuple(
                    job_id
                    for job_id in followups
                    if job_states.get(job_id) in QUEUED_JOB_STATES | RUNNING_JOB_STATES
                )
            except SlurmError as error:
                _LOGGER.warning('%s', error)
                job_states_known = False
    if machine.launcher == 'local':
        job_states = _local_states(machine=machine, study=study, jobs=jobs)
    reports = collect_states(
        study_directory=machine.output_root / study.name,
        runs=runs,
        jobs=jobs,
        job_states=job_states,
    )
    return Context(
        study=study,
        reference=reference,
        machine=machine,
        machine_path=machine_path,
        project_root=project_root,
        runs=runs,
        resolved={item.run.run_path: item for item in resolved},
        jobs=jobs,
        reports=reports,
        job_states_known=job_states_known,
        submission_in_progress=submission_in_progress(
            study_folder=study_folder(machine=machine, study=study)
        ),
        followup_live=bool(followup_live_job_ids),
        followup_live_job_ids=followup_live_job_ids,
    )


def _local_states(
    *, machine: MachineProfile, study: Study, jobs: list[JobRecord]
) -> dict[str, str]:
    live = local_job_states(study_folder=study_folder(machine=machine, study=study))
    # why: a local task outside a live launch is over, whatever its folder says
    ended = {
        f'{job.job_id}_{task.index}': LOCAL_ENDED
        for job, task in latest_tasks(jobs=jobs).values()
        if job.launcher == 'local'
    }
    return {**ended, **live}


def refresh_context(*, context: Context) -> Context:
    """The jobs and run states of a local context, read again for the same runs.

    Args:
        context: A context of a local machine.

    Returns:
        The context with fresh ``jobs`` and ``reports``.
    """
    jobs = read_jobs(
        path=study_folder(machine=context.machine, study=context.study) / 'submissions.jsonl'
    )
    reports = collect_states(
        study_directory=context.study_directory,
        runs=context.runs,
        jobs=jobs,
        job_states=_local_states(machine=context.machine, study=context.study, jobs=jobs),
    )
    return dataclasses.replace(context, jobs=jobs, reports=reports)


def _query_states(*, job_ids: list[str], ended_states: dict[str, str] | None) -> dict[str, str]:
    if ended_states is None:
        return query_job_states(job_ids=job_ids)
    queried = query_job_states(job_ids=[item for item in job_ids if item not in ended_states])
    # why: a failed query raised above, so the cache only ever holds states SLURM reported
    ended_states.update({item: state for item, state in queried.items() if job_ended(state)})
    return {**ended_states, **queried}


def load_context(  # noqa: PLR0913 # why: keyword-only public API; each option is documented below
    *,
    reference: str,
    machine_path: Path,
    selection: Selection,
    project_root: Path,
    read_only: bool = False,
    ended_states: dict[str, str] | None = None,
) -> Context:
    """The context of a study loaded from a ``module:attribute`` reference.

    Args:
        reference: The ``module:attribute`` reference of the study.
        machine_path: The machine profile file to load.
        selection: The grids and ``only`` matches to keep.
        project_root: The folder the command was called from; the jobs
            ``cd`` into it.
        read_only: See ``context_for``.
        ended_states: See ``context_for``.

    Returns:
        The context of the command.

    Raises:
        ValueError: If the reference is malformed or names no study.
        MachineProfileError: If the machine file does not hold a profile.
    """
    study = load_study(reference=reference)
    return context_for(
        study=study,
        machine_path=machine_path,
        selection=selection,
        project_root=project_root,
        reference=reference,
        read_only=read_only,
        ended_states=ended_states,
    )

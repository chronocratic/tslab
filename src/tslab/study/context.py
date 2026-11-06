"""What a study command reads before it does anything: the ``Context``."""

from collections.abc import Mapping, Sequence
import dataclasses
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from tslab.study.definition import (
    chained_grids,
    expand_runs,
    load_study,
    select_runs,
    Study,
    StudyRun,
    upstream_runs,
)
from tslab.study.machine import load_machine, MachineProfile
from tslab.study.paths import log_file_of, study_folder
from tslab.study.plan import check_machine, resolve_runs, ResolvedRun
from tslab.study.records import JobRecord, read_jobs, submission_in_progress
from tslab.study.reporting.chain import apply_chain_states
from tslab.study.reporting.state import collect_states, RunReport

if TYPE_CHECKING:
    from tslab.study.launchers.common.types import JobStates, QueuedJob

__all__ = [
    'Context',
    'Selection',
    'context_for',
    'expected_starts',
    'load_context',
    'refresh_context',
    'study_reports',
    'widen_context',
]


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
        upstream_reports: The reports used to decide dependencies of selected runs.
        job_states: The raw states and phases reported by the machine backend.
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
    resolved: dict[PurePosixPath, ResolvedRun]
    jobs: list[JobRecord]
    reports: list[RunReport]
    upstream_reports: Mapping[PurePosixPath, RunReport]
    job_states: 'JobStates'
    submission_in_progress: bool = False

    @property
    def study_directory(self) -> Path:
        """The ``<output_root>/<study>`` folder that holds the runs."""
        return self.machine.output_root / self.study.name

    def log_file(self, *, run: StudyRun, index: int) -> Path:
        """The scheduler log file of one task, ``.study/logs/<job name>_<index>.out``."""
        return log_file_of(
            folder=study_folder(machine=self.machine, study=self.study),
            job_name=self.study.job_name_of(run=run),
            index=index,
        )

    @property
    def followup_live(self) -> bool:
        """Whether a backend reports a live follow-up job."""
        return bool(self.job_states.followup_live_job_ids)

    @property
    def followup_live_job_ids(self) -> tuple[str, ...]:
        """The queued or running follow-up jobs reported by the backend."""
        return self.job_states.followup_live_job_ids


def expected_starts(*, context: Context) -> Mapping[str, 'QueuedJob'] | None:
    """The expected start and pending reason of the queued runs, keyed by job id.

    Args:
        context: The context to read the queued runs from.

    Returns:
        The expected start and pending reason of every queued run, keyed by its
        SLURM job id; ``None`` on a local machine, when no run is queued, or when
        the scheduler call fails (a failed call is logged as a warning).
    """
    # why: the launcher door imports Context, so this import must wait until calls
    from tslab.study.launchers.tables import backend_for  # noqa: PLC0415

    return backend_for(machine=context.machine).expected_starts(context=context)


def context_for(  # noqa: PLR0913 # why: keyword-only public API; each option is documented below
    *,
    study: Study,
    machine_path: Path,
    selection: Selection,
    project_root: Path,
    reference: str = '',
    read_only: bool = False,
    ended_states: dict[str, str] | None = None,
    local_cpus: int,
    environment: Mapping[str, str],
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
        local_cpus: The CPUs available to a local launch.
        environment: The process environment used to load the machine profile.

    Returns:
        The context of the study.

    Raises:
        MachineProfileError: If the machine file does not hold a profile.
        ValueError: If a setting path of the profile is not a field of a grid.
    """
    machine = load_machine(path=machine_path, lenient_env=read_only, environment=environment)
    # One set per command, so a dead override warns once, not once per check.
    warned: set[str] = set()
    check_machine(study=study, machine=machine, warned=warned)
    all_runs = expand_runs(study=study)
    runs = select_runs(study=study, runs=all_runs, grids=selection.grids, only=selection.only)
    resolved = resolve_runs(
        study=study, machine=machine, runs=runs, warned=warned, local_cpus=local_cpus
    )
    jobs = read_jobs(path=study_folder(machine=machine, study=study) / 'submissions.jsonl')
    # why: the launcher door imports Context, so this import must wait until calls
    from tslab.study.launchers.tables import backend_for  # noqa: PLC0415

    cache = {} if ended_states is None else ended_states
    job_states, new_cache = backend_for(machine=machine).job_states(
        machine=machine, study=study, jobs=jobs, cache=cache
    )
    if ended_states is not None:
        # why: study wait owns this cache across repeated context loads
        ended_states.update(new_cache)
    upstream_paths = _upstream_paths(study=study, runs=runs)
    state_paths = upstream_paths | {run.run_path for run in runs}
    state_runs = [run for run in all_runs if run.run_path in state_paths]
    state_reports = collect_states(
        study_directory=machine.output_root / study.name,
        runs=state_runs,
        jobs=jobs,
        job_states=job_states,
    )
    reports_by_path = {report.run.run_path: report for report in state_reports}
    upstream_reports = {
        path: reports_by_path[path] for path in upstream_paths if path in reports_by_path
    }
    reports = apply_chain_states(
        study=study,
        reports=[reports_by_path[run.run_path] for run in runs],
        upstream=upstream_reports,
        followup=machine.followup,
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
        upstream_reports=upstream_reports,
        job_states=job_states,
        submission_in_progress=submission_in_progress(
            study_folder=study_folder(machine=machine, study=study)
        ),
    )


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
    # why: the launcher door imports Context, so this import must wait until calls
    from tslab.study.launchers.tables import backend_for  # noqa: PLC0415

    job_states, _ = backend_for(machine=context.machine).job_states(
        machine=context.machine, study=context.study, jobs=jobs, cache={}
    )
    all_runs = expand_runs(study=context.study)
    upstream_paths = _upstream_paths(study=context.study, runs=context.runs)
    state_paths = upstream_paths | {run.run_path for run in context.runs}
    state_runs = [run for run in all_runs if run.run_path in state_paths]
    state_reports = collect_states(
        study_directory=context.study_directory, runs=state_runs, jobs=jobs, job_states=job_states
    )
    reports_by_path = {report.run.run_path: report for report in state_reports}
    upstream_reports = {
        path: reports_by_path[path] for path in upstream_paths if path in reports_by_path
    }
    reports = apply_chain_states(
        study=context.study,
        reports=[reports_by_path[run.run_path] for run in context.runs],
        upstream=upstream_reports,
        followup=context.machine.followup,
    )
    return dataclasses.replace(
        context,
        jobs=jobs,
        job_states=job_states,
        reports=reports,
        upstream_reports=upstream_reports,
    )


def load_context(  # noqa: PLR0913 # why: keyword-only public API; each option is documented below
    *,
    reference: str,
    machine_path: Path,
    selection: Selection,
    project_root: Path,
    read_only: bool = False,
    ended_states: dict[str, str] | None = None,
    local_cpus: int,
    environment: Mapping[str, str],
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
        local_cpus: See ``context_for``.
        environment: See ``context_for``.

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
        local_cpus=local_cpus,
        environment=environment,
    )


def _upstream_paths(*, study: Study, runs: list[StudyRun]) -> set[PurePosixPath]:
    """The unique upstream paths needed by selected chained runs."""
    if not chained_grids(study=study):
        return set()
    return {
        upstream.run_path
        for run in runs
        for after in study.grids[run.grid].after
        for upstream in upstream_runs(study=study, run=run, after=after)
    }


def study_reports(*, context: Context) -> dict[PurePosixPath, RunReport]:
    """The reports of every run of the study, chain states applied.

    Args:
        context: The context whose job records and states are read.

    Returns:
        The report per run path, selected or not.
    """
    raw = collect_states(
        study_directory=context.study_directory,
        runs=expand_runs(study=context.study),
        jobs=context.jobs,
        job_states=context.job_states,
    )
    reports = raw
    # why: a run behind two links needs the adjusted state of its direct upstream
    for _ in range(2):
        reports = apply_chain_states(
            study=context.study,
            reports=raw,
            upstream={report.run.run_path: report for report in reports},
            followup=context.machine.followup,
        )
    return {report.run.run_path: report for report in reports}


def widen_context(*, context: Context, reports: Sequence[RunReport]) -> Context:
    """The context with extra runs added, so commands can submit or cancel them.

    Args:
        context: The context to widen.
        reports: Reports of the runs to add; runs the context already holds are skipped.

    Returns:
        The context with the new runs, their resolved settings and reports, and the
        upstream reports their dependencies need.
    """
    # why: the launcher door imports Context, so this import must wait until calls
    from tslab.study.launchers.cpus import available_cpus  # noqa: PLC0415

    new = [report for report in reports if report.run.run_path not in context.resolved]
    if not new:
        return context
    runs = [report.run for report in new]
    resolved = resolve_runs(
        study=context.study, machine=context.machine, runs=runs, local_cpus=available_cpus()
    )
    everything = study_reports(context=context)
    upstream_paths = _upstream_paths(study=context.study, runs=runs)
    return dataclasses.replace(
        context,
        runs=[*context.runs, *runs],
        resolved={**context.resolved, **{item.run.run_path: item for item in resolved}},
        reports=[*context.reports, *new],
        upstream_reports={
            **context.upstream_reports,
            **{path: everything[path] for path in upstream_paths if path in everything},
        },
    )

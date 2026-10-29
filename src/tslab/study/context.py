"""What a study command reads before it does anything: the ``Context``."""

from dataclasses import dataclass
import logging
from pathlib import Path

from tslab.study.definition import expand_runs, select_runs, Study, StudyRun
from tslab.study.loading import load_study
from tslab.study.machine import load_machine, MachineProfile
from tslab.study.plan import check_machine, resolve_runs, ResolvedRun, study_folder
from tslab.study.records import JobRecord, read_jobs, submission_in_progress
from tslab.study.slurm import query_job_states, slurm_available, SlurmError
from tslab.study.state import (
    collect_states,
    followup_job_ids,
    QUEUED_JOB_STATES,
    RUNNING_JOB_STATES,
    RunReport,
    slurm_task_ids,
)

__all__ = ['Context', 'Selection', 'context_for', 'load_context']

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

    @property
    def study_directory(self) -> Path:
        """The ``<output_root>/<study>`` folder that holds the runs."""
        return self.machine.output_root / self.study.name


def context_for(  # noqa: PLR0913 # why: keyword-only public API; each option is documented below
    *,
    study: Study,
    machine_path: Path,
    selection: Selection,
    project_root: Path,
    reference: str = '',
    read_only: bool = False,
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

    Returns:
        The context of the study.

    Raises:
        MachineProfileError: If the machine file does not hold a profile.
        ValueError: If a setting path of the profile is not a field of a grid.
    """
    machine = load_machine(path=machine_path, lenient_env=read_only)
    check_machine(study=study, machine=machine)
    runs = select_runs(
        study=study, runs=expand_runs(study=study), grids=selection.grids, only=selection.only
    )
    resolved = resolve_runs(study=study, machine=machine, runs=runs)
    jobs = read_jobs(path=study_folder(machine=machine, study=study) / 'submissions.jsonl')
    job_states: dict[str, str] | None = None
    job_states_known = True
    followup_live = False
    if machine.launcher == 'slurm':
        job_states_known = slurm_available()
        if job_states_known:
            try:
                followups = followup_job_ids(jobs=jobs)
                job_states = query_job_states(job_ids=slurm_task_ids(jobs=jobs) + followups)
                followup_live = any(
                    job_states.get(job_id) in QUEUED_JOB_STATES | RUNNING_JOB_STATES
                    for job_id in followups
                )
            except SlurmError as error:
                _LOGGER.warning('%s', error)
                job_states_known = False
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
        followup_live=followup_live,
    )


def load_context(
    *,
    reference: str,
    machine_path: Path,
    selection: Selection,
    project_root: Path,
    read_only: bool = False,
) -> Context:
    """The context of a study loaded from a ``module:attribute`` reference.

    Args:
        reference: The ``module:attribute`` reference of the study.
        machine_path: The machine profile file to load.
        selection: The grids and ``only`` matches to keep.
        project_root: The folder the command was called from; the jobs
            ``cd`` into it.
        read_only: See ``context_for``.

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
    )

"""The ``tslab study`` commands: submit, and the shared pieces they build on.

Each command is a plain function over a ``Context``: the loaded study and
machine, the selected runs, the job records, and the state report. The CLI
dataclasses in ``tslab.cli.study`` parse the arguments and call these
functions, so the behavior is testable without the command line.
"""

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import logging
from pathlib import Path
import shlex
import time

from tslab.study.definition import expand_runs, select_runs, Study, StudyRun
from tslab.study.launch import launch, Launcher, LaunchReport, LocalLauncher, SlurmLauncher
from tslab.study.loading import load_study
from tslab.study.machine import load_machine, MachineProfile, SlurmSettings
from tslab.study.plan import (
    assign_indices,
    JobGroup,
    next_job_file,
    plan_jobs,
    resolve_runs,
    study_folder,
    validate_jobs,
)
from tslab.study.records import JobRecord, read_jobs
from tslab.study.slurm import array_spec, query_job_states, slurm_available
from tslab.study.state import collect_states, RunReport, slurm_task_ids, StudyState

__all__ = [
    'Context',
    'Selection',
    'load_context',
    'parse_states',
    'status',
    'submit',
    'submit_runs',
]

# the table columns, in display order
_TABLE_STATES: tuple[StudyState, ...] = tuple(StudyState)

_LOGGER = logging.getLogger(__name__)
_PROBLEMS_SHOWN = 20


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
        jobs: The job records of the study, in submission order.
        reports: The state of every selected run.
        job_states_known: Whether the SLURM states the reports use are real.
    """

    study: Study
    reference: str
    machine: MachineProfile
    machine_path: Path
    project_root: Path
    runs: list[StudyRun]
    jobs: list[JobRecord]
    reports: list[RunReport]
    job_states_known: bool


def load_context(
    *, reference: str, machine_path: Path, selection: Selection, project_root: Path
) -> Context:
    """The study, machine, runs, job records and states of one command.

    Args:
        reference: The ``module:attribute`` reference of the study.
        machine_path: The machine profile file to load.
        selection: The grids and ``only`` matches to keep.
        project_root: The folder the command was called from; the jobs
            ``cd`` into it.

    Returns:
        The context of the command.

    Raises:
        ValueError: If the reference is malformed or names no study, or a
            grid or ``only`` match is unknown to the study.
        MachineProfileError: If the machine file does not hold a profile.
    """
    study = load_study(reference=reference)
    machine = load_machine(path=machine_path)
    runs = select_runs(
        study=study, runs=expand_runs(study=study), grids=selection.grids, only=selection.only
    )
    folder = study_folder(machine=machine, study=study)
    jobs = read_jobs(path=folder / 'submissions.jsonl')
    job_states: dict[str, str] | None
    if machine.launcher == 'slurm':
        job_states_known = slurm_available()
        job_states = (
            query_job_states(job_ids=slurm_task_ids(jobs=jobs)) if job_states_known else None
        )
    else:
        job_states_known = True
        job_states = None
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
        jobs=jobs,
        reports=reports,
        job_states_known=job_states_known,
    )


def _launcher_of(
    *, machine: MachineProfile, project_root: Path, sleep: Callable[[float], None]
) -> Launcher:
    if machine.launcher == 'slurm':
        return SlurmLauncher(sleep=sleep)
    return LocalLauncher(project_root=project_root, env=machine.env)


def _resources_of(*, slurm: SlurmSettings | None) -> str:
    """The resource line of one group, for the dry run printout."""
    if slurm is None:
        return 'local'
    # why: the fields mirror SlurmSettings, named in display order
    parts = [f'partition={slurm.partition}', f'time={slurm.time}']
    if slurm.qos is not None:
        parts.append(f'qos={slurm.qos}')
    parts.append(f'cpus={slurm.cpus_per_task}')
    if slurm.mem_per_cpu is not None:
        parts.append(f'mem/cpu={slurm.mem_per_cpu}')
    if slurm.gpus:
        parts.append(f'gpus={slurm.gpus}')
    return ', '.join(parts)


def submit(
    *,
    context: Context,
    include_done: bool,
    dry_run: bool,
    extra_args: tuple[str, ...],
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Submit the runs that need submitting, on the context's machine.

    Queued and running runs are never submitted; done runs only when
    ``include_done``. Nothing is written when one run's arguments do not
    parse.

    Args:
        context: The context to submit from.
        include_done: Submit the done runs as well.
        dry_run: Write the job files and print the plan, submit nothing.
        extra_args: The pass-through arguments, appended to every run.
        sleep: What waits out the cluster's submission pause.

    Returns:
        ``0`` when everything submitted, or nothing needed submitting;
        ``1`` when a run does not parse or a submission failed.
    """
    to_submit = [
        report
        for report in context.reports
        if report.state not in (StudyState.QUEUED, StudyState.RUNNING)
        and (report.state is not StudyState.DONE or include_done)
    ]
    if not to_submit:
        _LOGGER.info('nothing to submit')
        return 0
    code, _ = submit_runs(
        context=context,
        reports=to_submit,
        dry_run=dry_run,
        extra_args={report.run.run_path: extra_args for report in to_submit},
        sleep=sleep,
    )
    return code


def parse_states(*, values: Sequence[str]) -> tuple[StudyState, ...]:
    """The state names of the ``--state`` flags, comma separated per flag.

    Args:
        values: The flag values, one per ``--state``.

    Returns:
        The states in the order the names were given.

    Raises:
        ValueError: If a name is not a state name; the six names are named.
    """
    states: list[StudyState] = []
    for value in values:
        for name in value.split(','):
            try:
                states.append(StudyState(name))
            except ValueError:
                allowed = ', '.join(state.value for state in StudyState)
                message = f'unknown state {name!r}; use one of: {allowed}'
                raise ValueError(message) from None
    return tuple(states)


def status(*, context: Context, states: tuple[StudyState, ...] = ()) -> int:
    """Print the state table of the study and one line per run not done.

    Args:
        context: The context to report on.
        states: When given, the listed lines are limited to these states.

    Returns:
        ``0``; the command only reads (spec §7).
    """
    if context.machine.launcher == 'slurm' and not context.job_states_known:
        _LOGGER.info(
            'SLURM commands not found: job states unknown; states come from run folders only'
        )
    counts: dict[str, Counter[StudyState]] = {}
    for report in context.reports:
        grid = counts.setdefault(report.run.grid, Counter())
        grid[report.state] += 1
    _LOGGER.info('  '.join(('grid', *(state.value for state in _TABLE_STATES), 'total')))
    for grid_name in sorted(counts):
        _LOGGER.info(_table_line(name=grid_name, grid=counts[grid_name]))
    total = sum(counts.values(), Counter[StudyState]())
    _LOGGER.info(_table_line(name='all', grid=total))
    for report in context.reports:
        if report.state is StudyState.DONE or (states and report.state not in states):
            continue
        job = report.job_id or '-'
        reason = report.reason or ''
        _LOGGER.info('%s %s %s %s', report.run.run_path, report.state.value, job, reason)
    limited = sum(
        1
        for report in context.reports
        if report.state is StudyState.INTERRUPTED
        and (
            report.job_state == 'TIMEOUT'
            or (report.reason or '').startswith('TrainingTimeLimitReached')
        )
    )
    if limited:
        _LOGGER.info(
            '%d runs stopped at a time limit: raise time in a machine override, '
            'or cap training with --trainer.max-training-time',
            limited,
        )
    return 0


def _table_line(*, name: str, grid: Counter[StudyState]) -> str:
    cells = [name, *(str(grid[state]) for state in _TABLE_STATES)]
    cells.append(str(sum(grid.values())))
    widths = [
        max(len(column), len(cell))
        for column, cell in zip(('grid', *(_TABLE_STATES), 'total'), cells, strict=True)
    ]
    return '  '.join(cell.rjust(width) for cell, width in zip(cells, widths, strict=True))


def submit_runs(  # noqa: PLR0913 # why: keyword-only public API; each option is documented below
    *,
    context: Context,
    reports: Sequence[RunReport],
    dry_run: bool,
    extra_args: Mapping[str, tuple[str, ...]],
    sleep: Callable[[float], None],
    begin: str | None = None,
) -> tuple[int, list[JobRecord]]:
    """Plan, validate and launch the given runs; the shared tail of the commands.

    A run keeps the array index of its first submission; its attempt is one
    past its latest recorded attempt.

    Args:
        context: The context to submit from.
        reports: The state reports of the runs to submit.
        dry_run: Write the job files and print the plan, submit nothing.
        extra_args: The run path to pass-through arguments.
        sleep: What waits out the cluster's submission pause.
        begin: A SLURM ``--begin`` for the submitted jobs.

    Returns:
        The exit code and the submitted run jobs; empty records on a dry
        run or when nothing was submitted.
    """
    runs = [report.run for report in reports]
    folder = study_folder(machine=context.machine, study=context.study)
    resolved = resolve_runs(study=context.study, machine=context.machine, runs=runs)
    indices = assign_indices(runs=runs, jobs=context.jobs)
    attempts = {report.run.run_path: report.attempt for report in reports}
    groups = plan_jobs(
        study=context.study,
        machine=context.machine,
        resolved=resolved,
        indices=indices,
        extra_args=extra_args,
        attempts=attempts,
        first_job_file=next_job_file(study_folder=folder),
    )
    problems = validate_jobs(study=context.study, groups=groups)
    if problems:
        _log_problems(problems=problems)
        return 1, []
    launcher = _launcher_of(machine=context.machine, project_root=context.project_root, sleep=sleep)
    report = launch(
        launcher=launcher,
        machine=context.machine,
        study=context.study,
        groups=groups,
        project_root=context.project_root,
        dry_run=dry_run,
        begin=begin,
    )
    return _report_launch(report=report, groups=groups)


def _log_problems(*, problems: list[tuple[str, str]]) -> None:
    for run_path, message in problems[:_PROBLEMS_SHOWN]:
        _LOGGER.error('%s: %s', run_path, message)
    hidden = len(problems) - _PROBLEMS_SHOWN
    if hidden > 0:
        _LOGGER.error('… and %d more', hidden)


def _report_launch(
    *, report: LaunchReport, groups: Sequence[JobGroup]
) -> tuple[int, list[JobRecord]]:
    names = {group.job_file: group.job_name for group in groups}
    if report.error is not None:
        _LOGGER.error('%s', report.error)
        _LOGGER.info(
            'not submitted: %s', ', '.join(names[group.job_file] for group in report.not_submitted)
        )
        return 1, list(report.submitted)
    if not report.submitted:
        for group in groups:
            _LOGGER.info(
                '%s: %d runs, %s',
                group.job_name,
                len(group.tasks),
                _resources_of(slurm=group.slurm),
            )
            if group.tasks:
                _LOGGER.info('%s', shlex.join(group.tasks[0].command))
        return 0, []
    for record in report.submitted:
        _LOGGER.info(
            'submitted %s as %s (%d runs, indices %s)',
            names.get(record.job_file, record.job_file),
            record.job_id,
            len(record.tasks),
            array_spec(indices=[task.index for task in record.tasks]),
        )
    return 0, list(report.submitted)

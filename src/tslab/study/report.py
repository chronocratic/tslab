"""What the study commands print: the state table, the launch plan, the problems."""

from collections import Counter
from collections.abc import Sequence
import logging
import shlex

from tslab.helpers.text import log_table
from tslab.jobs.slurm import array_spec
from tslab.study.context import Context
from tslab.study.definition import expand_runs, Study
from tslab.study.launch import LaunchReport
from tslab.study.machine import SlurmSettings
from tslab.study.plan import JobGroup
from tslab.study.records import JobRecord
from tslab.study.state import StudyState

__all__ = ['log_launch', 'log_problems', 'log_show', 'log_status', 'resources_of']

_LOGGER = logging.getLogger(__name__)
_PROBLEMS_SHOWN = 20

# the table columns, in display order
_TABLE_STATES: tuple[StudyState, ...] = tuple(StudyState)


def log_status(*, context: Context, states: tuple[StudyState, ...] = ()) -> None:
    """Log the state table of the study and one line per run not done.

    Args:
        context: The context to report on.
        states: When given, the listed lines are limited to these states.
    """
    if context.machine.launcher == 'slurm' and not context.job_states_known:
        _LOGGER.info(
            'SLURM commands not found: job states unknown; states come from run folders only'
        )
    counts: dict[str, Counter[StudyState]] = {}
    for report in context.reports:
        counts.setdefault(report.run.grid, Counter())[report.state] += 1
    total = sum(counts.values(), Counter[StudyState]())
    rows = [_table_row(name=name, grid=counts[name]) for name in sorted(counts)]
    rows.append(_table_row(name='all', grid=total))
    header = ('grid', *(state.value for state in _TABLE_STATES), 'total')
    log_table(logger=_LOGGER, header=header, rows=rows, right_align_numbers=True, total=True)
    listed = [
        report
        for report in context.reports
        if report.state is not StudyState.DONE and (not states or report.state in states)
    ]
    # why: a study that never ran has thousands of missing runs; the table already counts them
    if listed and not all(report.state is StudyState.MISSING for report in context.reports):
        lines = [
            ' '.join(
                part
                for part in (
                    report.run.run_path,
                    report.state.value,
                    None if report.job_id is None else f'job {report.job_id}',
                    report.reason,
                )
                if part is not None and part != ''
            )
            for report in listed
        ]
        for line in lines[:_PROBLEMS_SHOWN]:
            _LOGGER.info('  %s', line)
        hidden = len(lines) - _PROBLEMS_SHOWN
        if hidden > 0:
            _LOGGER.info('  … and %d more; narrow with --states, --grid or --only', hidden)
    limited = sum(
        1
        for report in context.reports
        if report.state is StudyState.INTERRUPTED
        and (report.job_state == 'TIMEOUT' or (report.reason or '').startswith('TrainingPaused:'))
    )
    if limited:
        _LOGGER.info(
            '%d runs stopped at a time limit: raise time in a machine override, '
            'or cap training with --trainer.max-training-time',
            limited,
        )


def log_show(*, study: Study) -> None:
    """Log the grids and axes of a study.

    Args:
        study: The study definition to show.

    Returns:
        None.
    """
    runs = expand_runs(study=study)
    rows = [
        (
            name,
            grid.short_name,
            str(sum(run.grid == name for run in runs)),
            ' '.join(f'{axis}={",".join(values)}' for axis, values in grid.axes.items()),
        )
        for name, grid in study.grids.items()
    ]
    log_table(logger=_LOGGER, header=('grid', 'short', 'runs', 'axes'), rows=rows)
    for name, grid in study.grids.items():
        _LOGGER.info('command of %s: %s', name, shlex.join(grid.command))
    _LOGGER.info('select with --grid <grid> or --only <axis>=<value>')


def _table_row(*, name: str, grid: Counter[StudyState]) -> list[str]:
    return [name, *(str(grid[state]) for state in _TABLE_STATES), str(sum(grid.values()))]


def resources_of(*, slurm: SlurmSettings | None) -> str:
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


def log_problems(*, problems: Sequence[tuple[str, str]]) -> None:
    """Log the planning problems, the first few of them.

    Args:
        problems: ``(run path, message)`` pairs.
    """
    for run_path, message in problems[:_PROBLEMS_SHOWN]:
        _LOGGER.error('%s: %s', run_path, message)
    hidden = len(problems) - _PROBLEMS_SHOWN
    if hidden > 0:
        _LOGGER.error('… and %d more', hidden)


def log_launch(*, report: LaunchReport, groups: Sequence[JobGroup]) -> tuple[int, list[JobRecord]]:
    """Log what one launch did and give the exit code and the submitted jobs.

    Args:
        report: The result of the launch.
        groups: The groups the launch was given.

    Returns:
        ``1`` with the jobs sent so far when the launch failed, else ``0``
        with the submitted jobs (none on a dry run).
    """
    names = {group.job_file: group.job_name for group in groups}
    if report.error is not None:
        _LOGGER.error('%s', report.error)
        _LOGGER.info(
            'not submitted: %s', ', '.join(names[group.job_file] for group in report.not_submitted)
        )
        return 1, list(report.submitted)
    if not report.submitted:
        rows = [
            (group.job_name, str(len(group.tasks)), resources_of(slurm=group.slurm))
            for group in groups
        ]
        log_table(logger=_LOGGER, header=('job', 'runs', 'resources'), rows=rows)
        for group in groups:
            if group.tasks:
                _LOGGER.info('first command of %s:', group.job_name)
                _LOGGER.info('  %s', shlex.join(group.tasks[0].command))
        return 0, []
    rows = [
        (
            names.get(record.job_file, str(record.job_file)),
            record.job_id,
            str(len(record.tasks)),
            array_spec(indices=[task.index for task in record.tasks]),
        )
        for record in report.submitted
    ]
    log_table(logger=_LOGGER, header=('job', 'job id', 'runs', 'indices'), rows=rows)
    return 0, list(report.submitted)

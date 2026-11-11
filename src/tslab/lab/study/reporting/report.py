"""What the study commands print: the state table, the launch plan, the problems."""

from collections import Counter
from collections.abc import Mapping, Sequence
import logging
from pathlib import PurePosixPath
import shlex

from tslab.lab.core.status import TRAINING_PAUSED_ERROR_PREFIX
from tslab.lab.core.text import format_index_ranges, log_table
from tslab.lab.enums import JobFlag, StudyState
from tslab.lab.study.context import Context
from tslab.lab.study.definition import expand_runs, Study
from tslab.lab.study.launchers.common.types import (
    LaunchReport,
    QueuedJob,
    Resources,
    SIGNAL_EXIT_BASE,
)
from tslab.lab.study.plan import JobGroup
from tslab.lab.study.records import JobRecord
from tslab.lab.study.reporting.chain import waits_for
from tslab.lab.study.reporting.state import RunReport

__all__ = ['log_launch', 'log_problems', 'log_show', 'log_status', 'resources_of']

_LOGGER = logging.getLogger(__name__)
_PROBLEMS_SHOWN = 20

# the table columns, in display order
_TABLE_STATES: tuple[StudyState, ...] = tuple(StudyState)


def log_status(
    *,
    context: Context,
    states: tuple[StudyState, ...] = (),
    starts: Mapping[str, QueuedJob] | None = None,
) -> None:
    """Log the state table of the study and one line per run not done.

    Args:
        context: The context to report on.
        states: When given, the listed lines are limited to these states.
        starts: When given, the lines of queued runs end with their expected
            start time and pending reason, as reported by the scheduler.
    """
    if not context.job_states.known:
        _LOGGER.info(
            'scheduler commands not found: job states unknown; states come from run folders only'
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
        _log_run_lines(context=context, reports=listed, starts=starts)
    limited = sum(
        1
        for report in context.reports
        if report.state is StudyState.INTERRUPTED
        and (
            JobFlag.TIME_LIMIT in report.job_flags
            or (report.reason or '').startswith(TRAINING_PAUSED_ERROR_PREFIX)
        )
    )
    if limited:
        _LOGGER.info(
            '%d runs stopped at a time limit: raise time in a machine override, '
            'or cap training with --trainer.max-training-time',
            limited,
        )
    held = sum(1 for report in context.reports if JobFlag.REQUEUE_HELD in report.job_flags)
    if held:
        _LOGGER.warning(
            '%d runs are held by the scheduler at its requeue limit and stay pending; '
            'the follow-up waits for them: run tslab study resubmit by hand',
            held,
        )


def _log_run_lines(
    *,
    reports: Sequence[RunReport],
    starts: Mapping[str, QueuedJob] | None,
    context: Context | None = None,
) -> None:
    """Log one line per listed run, the first few of them, then how many are hidden."""
    lines = [
        ' '.join(
            part
            for part in (
                report.run.run_path.as_posix(),
                report.state.value,
                None if report.job_id is None else f'job {report.job_id}',
                report.reason,
                _waits_for_suffix(context=context, report=report),
                _start_suffix(report=report, starts=starts),
            )
            if part is not None and part != ''
        )
        for report in reports
    ]
    for line in lines[:_PROBLEMS_SHOWN]:
        _LOGGER.info('  %s', line)
    hidden = len(lines) - _PROBLEMS_SHOWN
    if hidden > 0:
        _LOGGER.info('  … and %d more; narrow with --states, --grid or --only', hidden)


def _waits_for_suffix(*, context: Context | None, report: RunReport) -> str | None:
    """The upstream paths of a chained run that has not started or finished."""
    if context is None:
        return None
    paths = waits_for(context=context, report=report)
    return None if not paths else f'waits for {", ".join(paths)}'


def log_show(*, study: Study) -> None:
    """Log the grids and axes of a study.

    Args:
        study: The study definition to show.

    Returns:
        None.
    """
    if study.settings:
        _LOGGER.info('study settings:')
        for path, value in study.settings.items():
            _LOGGER.info('%s = %s', path, value)
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
        if grid.job_name != '{short_name}':
            _LOGGER.info('job name of %s: %s', name, grid.job_name)
    _LOGGER.info('select with --grid <grid> or --only <axis>=<value>')


def _table_row(*, name: str, grid: Counter[StudyState]) -> list[str]:
    return [name, *(str(grid[state]) for state in _TABLE_STATES), str(sum(grid.values()))]


def _start_suffix(*, report: RunReport, starts: Mapping[str, QueuedJob] | None) -> str | None:
    if starts is None or report.state is not StudyState.QUEUED or report.job_id is None:
        return None
    job = starts.get(report.job_id)
    if job is None:
        return None
    parts: list[str] = []
    if job.start is not None:
        parts.append(f'starts {job.start}')
    if job.reason is not None:
        waiting = 'waiting: ' if not job.stuck else 'will not start: '
        parts.append(f'{waiting}{job.reason}')
    return ', '.join(parts) if parts else None


def resources_of(*, resources: Resources) -> str:
    """The resource line of one group, for the dry run printout."""
    return resources.describe()


def log_problems(*, problems: Sequence[tuple[PurePosixPath, str]]) -> None:
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
        ``1`` with the jobs sent so far when the launch failed,
        ``128 + N`` with the submitted jobs when it was stopped by signal N,
        else ``0`` with the submitted jobs (none on a dry run).
    """
    names = {group.job_file: group.job_name for group in groups}
    if report.stopped_signal is not None:
        _LOGGER.info(
            'stopped by signal %d; resubmit continues the stopped runs', report.stopped_signal
        )
        return SIGNAL_EXIT_BASE + report.stopped_signal, list(report.submitted)
    if report.error is not None:
        _LOGGER.error('%s', report.error)
        _LOGGER.info(
            'not submitted: %s', ', '.join(names[group.job_file] for group in report.not_submitted)
        )
        return 1, list(report.submitted)
    if not report.submitted:
        rows = [
            (group.job_name, str(len(group.tasks)), resources_of(resources=group.resources))
            for group in groups
        ]
        log_table(logger=_LOGGER, header=('job', 'runs', 'resources'), rows=rows)
        for group in (group for group in groups if group.tasks):
            _LOGGER.info('first command of %s:', group.job_name)
            _LOGGER.info('  %s', shlex.join(group.tasks[0].command))
        return 0, []
    rows = [
        (
            names.get(record.job_file, str(record.job_file)),
            record.job_id,
            str(len(record.tasks)),
            format_index_ranges(indices=[task.index for task in record.tasks]),
        )
        for record in report.submitted
    ]
    log_table(logger=_LOGGER, header=('job', 'job id', 'runs', 'indices'), rows=rows)
    return 0, list(report.submitted)

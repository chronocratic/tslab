"""What the study commands print: the state table, the launch plan, the problems."""

from collections import Counter
from collections.abc import Sequence
import logging
import shlex

from tslab.core.text import format_table
from tslab.study.context import Context
from tslab.study.launch import LaunchReport
from tslab.study.machine import SlurmSettings
from tslab.study.plan import JobGroup
from tslab.study.records import JobRecord
from tslab.study.slurm import array_spec
from tslab.study.state import StudyState

__all__ = ['log_launch', 'log_problems', 'log_status', 'resources_of']

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
    for line in format_table(header=header, rows=rows, right_align_numbers=True):
        _LOGGER.info('%s', line)
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
        for group in groups:
            _LOGGER.info(
                '%s: %d runs, %s', group.job_name, len(group.tasks), resources_of(slurm=group.slurm)
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

"""Errors and recent log lines of failed or interrupted study runs."""

from collections import deque
from collections.abc import Mapping, Sequence
import logging
from pathlib import Path

from tslab.core.paths import ERROR_FILE, LOGS_DIRECTORY, require_path
from tslab.core.status import InstanceState
from tslab.jobs.slurm import format_slurm_memory, query_job_usage, SlurmError, SlurmJobUsage
from tslab.study.context import Context
from tslab.study.documents import run_entry
from tslab.study.plan import study_folder
from tslab.study.result_table import instance_directory
from tslab.study.state import RunReport, StudyState
from tslab.study.usage import _requested_memory

__all__ = ['collect_failures', 'failure_hint', 'log_failures', 'tail_lines']

_LOGGER = logging.getLogger(__name__)
_DEFAULT_STATES = (StudyState.FAILED, StudyState.INTERRUPTED)


def tail_lines(*, path: Path, lines: int) -> list[str] | None:
    """Read the last lines of a file, or return ``None`` when it is missing.

    Args:
        path: The file to read.
        lines: Maximum number of trailing lines.

    Returns:
        The trailing lines without line endings, or ``None`` when missing.
    """
    require_path(value=path, parameter='path')
    try:
        with path.open(encoding='utf-8', errors='replace') as file:
            return [line.rstrip('\r\n') for line in deque(file, maxlen=lines)]
    except FileNotFoundError:
        return None


def collect_failures(
    *, context: Context, states: tuple[StudyState, ...], lines: int
) -> list[dict[str, object]]:
    """Build failure details for the selected study runs.

    Args:
        context: The study context containing selected run reports.
        states: States to include; an empty tuple uses failed and interrupted.
        lines: Maximum number of trailing lines to read from each log.

    Returns:
        One failure document per matching run, in report order.
    """
    selected_states = states or _DEFAULT_STATES
    reports = [report for report in context.reports if report.state in selected_states]
    usage = _usage_of(context=context, reports=reports)
    return [
        _failure_entry(
            context=context,
            report=report,
            lines=lines,
            usage=usage.get(report.job_id) if report.job_id is not None else None,
        )
        for report in reports
    ]


def _usage_of(*, context: Context, reports: Sequence[RunReport]) -> dict[str, SlurmJobUsage]:
    """The accounting usage of the selected out-of-memory runs; no numbers on failure."""
    if context.machine.launcher != 'slurm':
        return {}
    oom_ids = [
        report.job_id
        for report in reports
        if report.job_state == 'OUT_OF_MEMORY' and report.job_id is not None
    ]
    if not oom_ids:
        return {}
    try:
        return query_job_usage(job_ids=oom_ids)
    except SlurmError as error:
        _LOGGER.warning('%s', error)
        return {}


def failure_hint(
    *, report: RunReport, usage: SlurmJobUsage | None, requested_memory: int | None
) -> str | None:
    """One hint for a failed or interrupted run; first match wins.

    Args:
        report: The state report for the run.
        usage: The accounting usage of the run, when queried.
        requested_memory: The memory requested by the run in bytes, when set.

    Returns:
        The hint text, or ``None`` when the error is in the code.
    """
    reason = report.reason or ''
    if reason.startswith('JobTimeLimitSignal:'):
        hint = 'stopped before the time limit; resubmit continues it, or raise time'
    elif reason.startswith('TrainingPaused:'):
        hint = (
            'training used its share of the job (job_time_share); '
            'resubmit continues it, or raise time to train longer per job'
        )
    elif report.job_state == 'OUT_OF_MEMORY':
        if usage is not None and usage.max_rss is not None and requested_memory is not None:
            hint = (
                'raise mem_per_cpu; peak '
                f'{format_slurm_memory(size=usage.max_rss)} '
                f'of {format_slurm_memory(size=requested_memory)} requested'
            )
        else:
            hint = 'raise mem_per_cpu'
    elif report.job_state == 'TIMEOUT':
        hint = (
            'raise time, or check that the timeout signal is on (signal_seconds), '
            'or lower job_time_share so the stages after training fit'
        )
    elif report.job_state in {'PREEMPTED', 'NODE_FAIL'}:
        hint = 'not caused by the code; resubmit'
    else:
        hint = None
    return hint


def _failure_entry(
    *, context: Context, report: RunReport, lines: int, usage: SlurmJobUsage | None
) -> dict[str, object]:
    run_directory = context.study_directory / report.run.run_path
    snapshot = report.folder.snapshot
    errors = []
    if snapshot is not None:
        for instance, record in snapshot.instances.items():
            if record.state is InstanceState.FAILED:
                error_file = (
                    instance_directory(run_directory=run_directory, instance_id=instance)
                    / ERROR_FILE
                )
                errors.append(
                    {
                        'instance': instance,
                        'error_file': error_file.resolve().as_posix(),
                        'error': record.error,
                    }
                )
    log_file = None
    log_tail = None
    if report.task is not None:
        grid = context.study.grids[report.run.grid]
        job_log_path = (
            study_folder(machine=context.machine, study=context.study).parent
            / LOGS_DIRECTORY
            / f'{grid.short_name}_{report.task.index}.out'
        )
        log_file = job_log_path.resolve().as_posix()
        log_tail = tail_lines(path=job_log_path, lines=lines)
    run_log_path = run_directory / LOGS_DIRECTORY / 'run.log'
    entry = run_entry(report=report, context=context)
    entry.update(
        {
            'errors': errors,
            'log_file': log_file,
            'log_tail': log_tail,
            'run_log': run_log_path.resolve().as_posix(),
            'run_log_tail': tail_lines(path=run_log_path, lines=lines),
            'hint': failure_hint(
                report=report,
                usage=usage,
                requested_memory=_requested_memory(
                    slurm=context.resolved[report.run.run_path].slurm
                ),
            ),
        }
    )
    return entry


def log_failures(*, failures: Sequence[Mapping[str, object]]) -> None:
    """Log status, hints, errors, and log tails for each failed run.

    Args:
        failures: Failure entries returned by ``collect_failures``.

    Returns:
        None.
    """
    for failure in failures:
        _log_status_line(failure=failure)
        hint = failure.get('hint')
        if isinstance(hint, str) and hint:
            _LOGGER.info('hint: %s', hint)
        errors = failure.get('errors')
        if isinstance(errors, Sequence):
            for error in errors:
                if isinstance(error, Mapping):
                    _LOGGER.error(
                        '%s failed; full error: %s', error.get('instance'), error.get('error_file')
                    )
                    if error.get('error') is not None:
                        _LOGGER.error('%s', error.get('error'))
        _log_tail(path=failure.get('log_file'), tail=failure.get('log_tail'))
        _log_tail(path=failure.get('run_log'), tail=failure.get('run_log_tail'))


def _log_status_line(*, failure: Mapping[str, object]) -> None:
    parts: list[str] = []
    for name in ('run_path', 'state'):
        value = failure.get(name)
        if isinstance(value, str):
            parts.append(value)
    job_id = failure.get('job_id')
    if isinstance(job_id, str):
        parts.append(f'job {job_id}')
    reason = failure.get('reason')
    if isinstance(reason, str) and reason:
        parts.append(reason)
    _LOGGER.info('%s', ' '.join(parts))


def _log_tail(*, path: object, tail: object) -> None:
    if not isinstance(path, str):
        return
    if tail is None:
        _LOGGER.info('%s: missing', path)
        return
    if isinstance(tail, Sequence):
        _LOGGER.info('--- %s (last %d lines)', path, len(tail))
        for line in tail:
            _LOGGER.info('%s', line)

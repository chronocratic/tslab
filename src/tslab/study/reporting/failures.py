"""Errors and recent log lines of failed or interrupted study runs."""

from collections.abc import Mapping, Sequence
import logging

from tslab.core.paths import ERROR_FILE, LOGS_DIRECTORY
from tslab.enums import InstanceState, StudyState
from tslab.jobs.local import tail_lines
from tslab.study.context import Context
from tslab.study.launchers.common.protocols import Backend
from tslab.study.launchers.common.types import JobUsage, Resources
from tslab.study.launchers.tables import backend_for
from tslab.study.paths import instance_directory
from tslab.study.reporting.documents import run_entry
from tslab.study.reporting.state import RunReport

__all__ = ['collect_failures', 'failure_hint', 'log_failures']

_LOGGER = logging.getLogger(__name__)
_DEFAULT_STATES = (StudyState.FAILED, StudyState.INTERRUPTED)


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

    usage = backend_for(machine=context.machine).failure_usage(context=context, reports=reports)
    return [
        _failure_entry(
            context=context,
            report=report,
            lines=lines,
            usage=usage.get(report.job_id) if report.job_id is not None else None,
        )
        for report in reports
    ]


_REASON_HINTS = (
    ('JobTimeLimitSignal:', 'stopped before the time limit; resubmit continues it, or raise time'),
    (
        'TrainingPaused:',
        (
            'training used its share of the job (job_time_share); '
            'resubmit continues it, or raise time to train longer per job'
        ),
    ),
)


def failure_hint(
    *, report: RunReport, usage: JobUsage | None, backend: Backend, resources: Resources
) -> str | None:
    """One hint for a failed or interrupted run; the reason of the run comes first.

    Args:
        report: The state report for the run.
        usage: The accounting usage of the run, when queried.
        backend: The backend of the machine, for the hint of the scheduler state.
        resources: The resources of the run.

    Returns:
        The hint text, or ``None`` when the error is in the code.
    """
    reason = report.reason or ''
    reason_hint = next((hint for prefix, hint in _REASON_HINTS if reason.startswith(prefix)), None)
    if reason_hint is not None:
        return reason_hint
    return backend.failure_hint(report=report, usage=usage, resources=resources)


def _failure_entry(
    *, context: Context, report: RunReport, lines: int, usage: JobUsage | None
) -> dict[str, object]:
    run_directory = context.study_directory / report.run.run_path
    snapshot = report.folder.snapshot
    failed = {} if snapshot is None else snapshot.instances
    errors = [
        {
            'instance': instance,
            'error_file': (
                (instance_directory(run_directory=run_directory, instance_id=instance) / ERROR_FILE)
                .resolve()
                .as_posix()
            ),
            'error': record.error,
        }
        for instance, record in failed.items()
        if record.state is InstanceState.FAILED
    ]
    log_file = None
    log_tail = None
    if report.task is not None:
        job_log_path = context.log_file(run=report.run, index=report.task.index)
        log_file = job_log_path.resolve().as_posix()
        log_tail = tail_lines(path=job_log_path, lines=lines)
    run_log_path = run_directory / LOGS_DIRECTORY / 'run.log'
    entry = run_entry(report=report, context=context)
    resources = context.resolved[report.run.run_path].resources
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
                backend=backend_for(machine=context.machine),
                resources=resources,
            ),
            'exit_code': None if usage is None else usage.exit_code,
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
        for error in errors if isinstance(errors, Sequence) else ():
            _log_error(error=error)
        _log_tail(path=failure.get('log_file'), tail=failure.get('log_tail'))
        _log_tail(path=failure.get('run_log'), tail=failure.get('run_log_tail'))


def _log_error(*, error: object) -> None:
    """Log one failed instance of a failure entry: where its error is, and the error."""
    if not isinstance(error, Mapping):
        return
    _LOGGER.error('%s failed; full error: %s', error.get('instance'), error.get('error_file'))
    if error.get('error') is not None:
        _LOGGER.error('%s', error.get('error'))


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
    exit_code = failure.get('exit_code')
    if isinstance(exit_code, str):
        parts.append(f'exit {exit_code}')
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

"""JSON documents for study commands."""

from tslab.study.context import Context
from tslab.study.outcome import counts_live
from tslab.study.state import RunReport, StudyState

__all__ = ['run_entry', 'status_document']


def run_entry(*, report: RunReport, context: Context) -> dict[str, object]:
    """The JSON fields for one selected run.

    Args:
        report: The state report for the run.
        context: The study context containing its folder and live-work state.

    Returns:
        The run entry with absolute paths.
    """
    reason = report.reason
    if counts_live(report=report, context=context):
        if report.state is StudyState.MISSING:
            reason = 'submission in progress'
        elif report.state is StudyState.INTERRUPTED:
            job_id = context.followup_live_job_ids[0] if context.followup_live_job_ids else None
            reason = 'auto retry queued' if job_id is None else f'auto retry queued (job {job_id})'
    return {
        'run_path': report.run.run_path,
        'grid': report.run.grid,
        'values': dict(report.run.values),
        'state': report.state.value,
        'reason': reason,
        'job_id': report.job_id,
        'job_state': report.job_state,
        'attempt': report.attempt,
        'directory': (context.study_directory / report.run.run_path).resolve().as_posix(),
    }


def status_document(*, context: Context, exit_code: int) -> dict[str, object]:
    """The JSON document for the status of selected study runs.

    Args:
        context: The study context containing selected runs and machine details.
        exit_code: The status code to report, regardless of CLI opt-in.

    Returns:
        The versioned status document.
    """
    counts: dict[str, dict[str, int]] = {}
    for grid in sorted({report.run.grid for report in context.reports}):
        counts[grid] = _empty_counts()
    counts['all'] = _empty_counts()
    for report in context.reports:
        counts[report.run.grid][report.state.value] += 1
        counts['all'][report.state.value] += 1
        counts[report.run.grid]['total'] += 1
        counts['all']['total'] += 1
    return {
        'schema': 'tslab.study-status/1',
        'study': context.reference or context.study.name,
        'machine': context.machine_path.resolve().as_posix(),
        'study_directory': context.study_directory.resolve().as_posix(),
        'job_states_known': context.job_states_known,
        'exit_code': exit_code,
        'counts': counts,
        'runs': [run_entry(report=report, context=context) for report in context.reports],
    }


def _empty_counts() -> dict[str, int]:
    return {**{state.value: 0 for state in StudyState}, 'total': 0}

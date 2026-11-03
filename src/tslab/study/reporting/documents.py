"""JSON documents for study commands."""

from collections.abc import Callable, Mapping, Sequence
import shlex
from types import MappingProxyType

from tslab.core.text import format_index_ranges
from tslab.enums import StudyState
from tslab.study.context import Context
from tslab.study.definition import expand_runs, Study
from tslab.study.launchers import LaunchReport, QueuedJob
from tslab.study.plan import JobGroup
from tslab.study.records import JobRecord
from tslab.study.reporting.outcome import counts_live
from tslab.study.reporting.report import resources_of
from tslab.study.reporting.state import RunReport

__all__ = ['cancel_document', 'launch_document', 'run_entry', 'show_document', 'status_document']


def _submission_reason(*, context: Context) -> str:  # noqa: ARG001 # why: the table signature
    return 'submission in progress'


def _retry_reason(*, context: Context) -> str:
    job_id = context.followup_live_job_ids[0] if context.followup_live_job_ids else None
    return 'auto retry queued' if job_id is None else f'auto retry queued (job {job_id})'


#: The reason of a run that counts as live work, by its state.
_LIVE_REASONS: Mapping[StudyState, Callable[..., str]] = MappingProxyType(
    {StudyState.MISSING: _submission_reason, StudyState.INTERRUPTED: _retry_reason}
)


def run_entry(
    *, report: RunReport, context: Context, starts: Mapping[str, QueuedJob] | None = None
) -> dict[str, object]:
    """The JSON fields for one selected run.

    Args:
        report: The state report for the run.
        context: The study context containing its folder and live-work state.
        starts: When given, the entry gains ``expected_start``, ``pending_reason``
            and ``pending_stuck`` fields for the job of the run.

    Returns:
        The run entry with absolute paths.
    """
    reason = report.reason
    live_reason = _LIVE_REASONS.get(report.state)
    if live_reason is not None and counts_live(report=report, context=context):
        reason = live_reason(context=context)
    entry: dict[str, object] = {
        'run_path': report.run.run_path.as_posix(),
        'grid': report.run.grid,
        'values': dict(report.run.values),
        'state': report.state.value,
        'reason': reason,
        'job_id': report.job_id,
        'job_state': report.job_state,
        'attempt': report.attempt,
        'directory': (context.study_directory / report.run.run_path).resolve().as_posix(),
    }
    if starts is not None:
        job = starts.get(report.job_id) if report.job_id is not None else None
        entry['expected_start'] = None if job is None else job.start
        entry['pending_reason'] = None if job is None else job.reason
        entry['pending_stuck'] = False if job is None else job.stuck
    return entry


def status_document(
    *, context: Context, exit_code: int, starts: Mapping[str, QueuedJob] | None = None
) -> dict[str, object]:
    """The JSON document for the status of selected study runs.

    Args:
        context: The study context containing selected runs and machine details.
        exit_code: The status code to report, regardless of CLI opt-in.
        starts: When given, every run entry gains ``expected_start``,
            ``pending_reason`` and ``pending_stuck`` fields for its job.

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
        'job_states_known': context.job_states.known,
        'exit_code': exit_code,
        'counts': counts,
        'runs': [
            run_entry(report=report, context=context, starts=starts) for report in context.reports
        ],
    }


def show_document(*, study: Study) -> dict[str, object]:
    """The grids, axes, and run counts of a study.

    Args:
        study: The study definition to describe.

    Returns:
        The versioned study-show document.
    """
    runs = expand_runs(study=study)
    grids = [
        {
            'name': name,
            'short_name': grid.short_name,
            'command': list(grid.command),
            'axes': {axis: list(values) for axis, values in grid.axes.items()},
            'excludes': len(grid.exclude),
            'overrides': len(grid.overrides),
            'runs': sum(run.grid == name for run in runs),
        }
        for name, grid in study.grids.items()
    ]
    return {
        'schema': 'tslab.study-show/1',
        'study': study.name,
        'grids': grids,
        'total_runs': len(runs),
    }


def cancel_document(
    *, cancelled: Sequence[str], dry_run: bool, exit_code: int, signal: str | None = None
) -> dict[str, object]:
    """The JSON document for a study cancellation.

    Args:
        cancelled: The job ids cancelled or selected for cancellation.
        dry_run: Whether cancellation was only previewed.
        exit_code: The command exit code.
        signal: The signal sent to the running jobs, or ``None`` when the
            jobs were cancelled or left untouched.

    Returns:
        The versioned study-cancel document.
    """
    return {
        'schema': 'tslab.study-cancel/1',
        'cancelled': list(cancelled),
        'dry_run': dry_run,
        'signal': signal,
        'exit_code': exit_code,
    }


def _launch_jobs(
    *, report: LaunchReport | None, groups: Sequence[JobGroup], dry_run: bool
) -> list[dict[str, object]]:
    """The job entries of a launch: every group on a dry run, else the submitted ones."""
    if report is None:
        return []
    records = {record.job_file: record for record in report.submitted}
    return [
        _job_entry(group=group, record=records.get(group.job_file), dry_run=dry_run)
        for group in groups
        if dry_run or group.job_file in records
    ]


def _job_entry(*, group: JobGroup, record: JobRecord | None, dry_run: bool) -> dict[str, object]:
    """The entry of one job group in a launch document."""
    return {
        'job_name': group.job_name,
        'job_id': None if dry_run or record is None else record.job_id,
        'runs': len(group.tasks),
        'indices': format_index_ranges(indices=[task.index for task in group.tasks]),
        'resources': resources_of(resources=group.resources),
        'first_command': shlex.join(group.tasks[0].command) if group.tasks else None,
    }


def launch_document(
    *,
    report: LaunchReport | None,
    groups: Sequence[JobGroup],
    dry_run: bool,
    cancelled: Sequence[str],
    exit_code: int,
) -> dict[str, object]:
    """The JSON document for a submit or resubmit launch.

    Args:
        report: The result of the launch, or ``None`` when nothing was launched.
        groups: The groups planned for this launch.
        dry_run: Whether the launch only wrote its plan.
        cancelled: The cancelled job ids, for resubmit.
        exit_code: The command exit code.

    Returns:
        The versioned study-launch document.
    """
    jobs = _launch_jobs(report=report, groups=groups, dry_run=dry_run)
    return {
        'schema': 'tslab.study-launch/1',
        'dry_run': dry_run,
        'jobs': jobs,
        'cancelled': list(cancelled),
        'exit_code': exit_code,
    }


def _empty_counts() -> dict[str, int]:
    return {**{state.value: 0 for state in StudyState}, 'total': 0}

"""JSON documents for study commands."""

from collections.abc import Mapping, Sequence
import shlex

from tslab.jobs.slurm import array_spec
from tslab.study.context import Context
from tslab.study.definition import expand_runs, Study
from tslab.study.launchers import LaunchReport
from tslab.study.outcome import counts_live
from tslab.study.plan import JobGroup
from tslab.study.report import resources_of
from tslab.study.state import RunReport, StudyState

__all__ = ['cancel_document', 'launch_document', 'run_entry', 'show_document', 'status_document']


def run_entry(
    *, report: RunReport, context: Context, starts: Mapping[str, str | None] | None = None
) -> dict[str, object]:
    """The JSON fields for one selected run.

    Args:
        report: The state report for the run.
        context: The study context containing its folder and live-work state.
        starts: When given, the entry gains an ``expected_start`` field with the
            expected start of the job of the run, or ``None``.

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
    entry: dict[str, object] = {
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
    if starts is not None:
        entry['expected_start'] = starts.get(report.job_id) if report.job_id is not None else None
    return entry


def status_document(
    *, context: Context, exit_code: int, starts: Mapping[str, str | None] | None = None
) -> dict[str, object]:
    """The JSON document for the status of selected study runs.

    Args:
        context: The study context containing selected runs and machine details.
        exit_code: The status code to report, regardless of CLI opt-in.
        starts: When given, every run entry gains an ``expected_start`` field
            with the expected start of its job, or ``null``.

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
    jobs = []
    if report is not None:
        records = {record.job_file: record for record in report.submitted}
        for group in groups:
            record = records.get(group.job_file)
            if not dry_run and record is None:
                continue
            jobs.append(
                {
                    'job_name': group.job_name,
                    'job_id': None if dry_run else record.job_id if record is not None else None,
                    'runs': len(group.tasks),
                    'indices': array_spec(indices=[task.index for task in group.tasks]),
                    'resources': resources_of(slurm=group.slurm),
                    'first_command': shlex.join(group.tasks[0].command) if group.tasks else None,
                }
            )
    return {
        'schema': 'tslab.study-launch/1',
        'dry_run': dry_run,
        'jobs': jobs,
        'cancelled': list(cancelled),
        'exit_code': exit_code,
    }


def _empty_counts() -> dict[str, int]:
    return {**{state.value: 0 for state in StudyState}, 'total': 0}

"""The post-pass that derives chained-grid states and wait targets."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from tslab.enums import AfterVerdict, JobFlag, StudyState
from tslab.study.chain import after_verdict
from tslab.study.definition import upstream_runs

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import PurePosixPath

    from tslab.study.context import Context
    from tslab.study.definition import Study
    from tslab.study.machine.profile import FollowupSettings
    from tslab.study.reporting.state import RunReport

__all__ = ['apply_chain_states', 'waits_for']


def apply_chain_states(
    *,
    study: Study,
    reports: Sequence[RunReport],
    upstream: Mapping[PurePosixPath, RunReport],
    followup: FollowupSettings,
) -> list[RunReport]:
    """Apply dependency-specific state changes to selected run reports.

    Args:
        study: The study whose grids declare dependencies.
        reports: Reports to update, usually the selected runs.
        upstream: Reports needed to judge downstream dependencies.
        followup: Retry settings that keep eligible interruptions open.

    Returns:
        Updated report list in input order.
    """
    return [
        _chain_report(study=study, report=report, upstream=upstream, followup=followup)
        for report in reports
    ]


def waits_for(*, context: Context, report: RunReport) -> tuple[str, ...]:
    """The matching upstream run paths for a non-final chained report.

    Args:
        context: The context with collected upstream reports.
        report: The report shown to the user.

    Returns:
        Upstream paths for queued, waiting or missing downstream runs.
    """
    if report.state not in (StudyState.QUEUED, StudyState.WAITING, StudyState.MISSING):
        return ()
    grid = context.study.grids[report.run.grid]
    paths = (
        upstream.run_path.as_posix()
        for after in grid.after
        for upstream in upstream_runs(study=context.study, run=report.run, after=after)
        if upstream.run_path in context.upstream_reports
    )
    return tuple(dict.fromkeys(paths))


def _chain_report(
    *,
    study: Study,
    report: RunReport,
    upstream: Mapping[PurePosixPath, RunReport],
    followup: FollowupSettings,
) -> RunReport:
    """Apply the first applicable dependency state rule to one report."""
    if report.state is StudyState.QUEUED and JobFlag.DEPENDENCY_PENDING in report.job_flags:
        return replace(report, state=StudyState.WAITING)
    if JobFlag.DEPENDENCY_NEVER in report.job_flags and report.folder.snapshot is None:
        return replace(report, state=StudyState.DEPENDENCY_FAILED)
    if (
        report.folder.snapshot is not None
        or report.state not in (StudyState.MISSING, StudyState.FAILED)
        or not study.grids[report.run.grid].after
    ):
        return report
    verdict = after_verdict(study=study, run=report.run, reports=upstream, followup=followup)
    if verdict is AfterVerdict.NEVER:
        return replace(report, state=StudyState.DEPENDENCY_FAILED)
    if _job_was_cancelled(report=report):
        return replace(report, state=StudyState.MISSING)
    return report


def _job_was_cancelled(*, report: RunReport) -> bool:
    """Whether the backend reports a cancelled task."""
    return JobFlag.CANCELLED in report.job_flags

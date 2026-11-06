"""Condition logic shared by chained-grid launchers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tslab.enums import AfterCondition, AfterVerdict, ChainedRetry, Satisfy, StudyState
from tslab.study.definition import upstream_runs

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import PurePosixPath

    from tslab.study.definition import After, Study, StudyRun
    from tslab.study.machine.profile import FollowupSettings
    from tslab.study.reporting.state import RunReport

__all__ = ['after_verdict', 'upstream_verdict']


def upstream_verdict(
    *, condition: AfterCondition, state: StudyState, retry_left: bool
) -> AfterVerdict:
    """Judge one upstream state against a dependency condition.

    Args:
        condition: The condition declared by the downstream grid.
        state: The current upstream run state.
        retry_left: Whether the upstream run can still be retried.

    Returns:
        Whether the condition is met, impossible or still open.
    """
    if condition is AfterCondition.OK:
        return _ok_verdict(state=state, retry_left=retry_left)
    if condition is AfterCondition.ANY:
        return _any_verdict(state=state)
    if condition is AfterCondition.FAILED:
        return _failed_verdict(state=state, retry_left=retry_left)
    return _started_verdict(state=state)


def after_verdict(
    *,
    study: Study,
    run: StudyRun,
    reports: Mapping[PurePosixPath, RunReport],
    followup: FollowupSettings | None = None,
) -> AfterVerdict:
    """Combine every declared upstream condition for one downstream run.

    Args:
        study: The study containing both grids.
        run: The downstream run to judge.
        reports: The known upstream reports, keyed by run path.
        followup: Retry settings used to keep retryable interruptions open.

    Returns:
        The combined dependency verdict; every ``After`` entry combines with all.
    """
    grid = study.grids[run.grid]
    verdicts = [
        _entry_verdict(study=study, run=run, after=after, reports=reports, followup=followup)
        for after in grid.after
    ]
    if AfterVerdict.NEVER in verdicts:
        return AfterVerdict.NEVER
    if all(verdict is AfterVerdict.MET for verdict in verdicts):
        return AfterVerdict.MET
    return AfterVerdict.OPEN


def _entry_verdict(
    *,
    study: Study,
    run: StudyRun,
    after: After,
    reports: Mapping[PurePosixPath, RunReport],
    followup: FollowupSettings | None,
) -> AfterVerdict:
    """Combine matched runs according to one entry's ``satisfy`` choice."""
    matched = upstream_runs(study=study, run=run, after=after)
    if not matched:
        return AfterVerdict.OPEN
    verdicts = [
        _report_verdict(
            report=reports.get(upstream.run_path), condition=after.condition, followup=followup
        )
        for upstream in matched
    ]
    if after.satisfy is Satisfy.ALL:
        return _all_verdict(verdicts=verdicts)
    return _any_entry_verdict(verdicts=verdicts)


def _all_verdict(*, verdicts: list[AfterVerdict]) -> AfterVerdict:
    """The verdict when every matched upstream must meet its condition."""
    if AfterVerdict.NEVER in verdicts:
        return AfterVerdict.NEVER
    return AfterVerdict.MET if all(v is AfterVerdict.MET for v in verdicts) else AfterVerdict.OPEN


def _any_entry_verdict(*, verdicts: list[AfterVerdict]) -> AfterVerdict:
    """The verdict when one matched upstream must meet its condition."""
    if AfterVerdict.MET in verdicts:
        return AfterVerdict.MET
    if all(verdict is AfterVerdict.NEVER for verdict in verdicts):
        return AfterVerdict.NEVER
    return AfterVerdict.OPEN


def _report_verdict(
    *, report: RunReport | None, condition: AfterCondition, followup: FollowupSettings | None
) -> AfterVerdict:
    """Judge one report, treating an uncollected report as open."""
    if report is None:
        return AfterVerdict.OPEN
    return upstream_verdict(
        condition=condition,
        state=report.state,
        retry_left=_retry_left(report=report, followup=followup),
    )


def _retry_left(*, report: RunReport, followup: FollowupSettings | None) -> bool:
    """Whether chained retry settings permit another attempt for this report."""
    if followup is None:
        return False
    return (
        followup.chained_retry is not ChainedRetry.MANUAL and report.attempt < followup.max_attempts
    )


def _ok_verdict(*, state: StudyState, retry_left: bool) -> AfterVerdict:
    """The verdict for an ``OK`` dependency."""
    if state is StudyState.DONE:
        return AfterVerdict.MET
    if state in (StudyState.FAILED, StudyState.DEPENDENCY_FAILED):
        return AfterVerdict.NEVER
    if state is StudyState.INTERRUPTED:
        return AfterVerdict.OPEN if retry_left else AfterVerdict.NEVER
    return AfterVerdict.OPEN


def _any_verdict(*, state: StudyState) -> AfterVerdict:
    """The verdict for an ``ANY`` dependency."""
    if state in (
        StudyState.DONE,
        StudyState.FAILED,
        StudyState.INTERRUPTED,
        StudyState.DEPENDENCY_FAILED,
    ):
        return AfterVerdict.MET
    return AfterVerdict.OPEN


def _failed_verdict(*, state: StudyState, retry_left: bool) -> AfterVerdict:
    """The verdict for a ``FAILED`` dependency."""
    if state is StudyState.DONE:
        return AfterVerdict.NEVER
    if state in (StudyState.FAILED, StudyState.DEPENDENCY_FAILED):
        return AfterVerdict.MET
    if state is StudyState.INTERRUPTED:
        return AfterVerdict.OPEN if retry_left else AfterVerdict.MET
    return AfterVerdict.OPEN


def _started_verdict(*, state: StudyState) -> AfterVerdict:
    """The verdict for a ``STARTED`` dependency."""
    if state is StudyState.DEPENDENCY_FAILED:
        return AfterVerdict.NEVER
    if state in (StudyState.RUNNING, StudyState.DONE, StudyState.INTERRUPTED, StudyState.FAILED):
        return AfterVerdict.MET
    return AfterVerdict.OPEN

"""Condition logic shared by chained-grid launchers."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, TYPE_CHECKING

from tslab.lab.enums import AfterCondition, AfterVerdict, ChainedRetry, Satisfy, StudyState
from tslab.lab.study.definition.after import upstream_runs

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence
    from pathlib import PurePosixPath

    from tslab.lab.study.definition.after import After
    from tslab.lab.study.definition.definition import Study, StudyRun
    from tslab.lab.study.machine.profile import FollowupSettings
    from tslab.lab.study.reporting.state import RunReport

__all__ = ['after_verdict', 'downstream_of', 'retry_left', 'upstream_verdict']


@dataclass(frozen=True, kw_only=True)
class _VerdictFacts:
    """State and timing facts used by one dependency verdict.

    Attributes:
        state: The current upstream run state.
        retry_left: Whether the upstream run can still be retried.
        delay_minutes: The required delay after the upstream starts.
        started_at: The upstream start time in UTC Unix seconds, when known.
        now: The current time in UTC Unix seconds, when known.
    """

    state: StudyState
    retry_left: bool
    delay_minutes: int
    started_at: float | None
    now: float | None


class _VerdictStrategy(Protocol):
    def __call__(self, *, facts: _VerdictFacts) -> AfterVerdict: ...


def upstream_verdict(  # noqa: PLR0913 # why: state and retry timing facts are explicit inputs
    *,
    condition: AfterCondition,
    state: StudyState,
    retry_left: bool,
    delay_minutes: int = 0,
    started_at: float | None = None,
    now: float | None = None,
) -> AfterVerdict:
    """Judge one upstream state against a dependency condition.

    Args:
        condition: The condition declared by the downstream grid.
        state: The current upstream run state.
        retry_left: Whether the upstream run can still be retried.
        delay_minutes: Minutes that must pass after the upstream job started before
            ``STARTED`` is met.
        started_at: The upstream job start time in UTC Unix seconds, when known.
        now: The current time in UTC Unix seconds, passed by the caller.

    Returns:
        Whether the condition is met, impossible or still open.
    """
    facts = _VerdictFacts(
        state=state,
        retry_left=retry_left,
        delay_minutes=delay_minutes,
        started_at=started_at,
        now=now,
    )
    return _VERDICT_BY_CONDITION[condition](facts=facts)


def after_verdict(
    *,
    study: Study,
    run: StudyRun,
    reports: Mapping[PurePosixPath, RunReport],
    followup: FollowupSettings | None = None,
    now: float | None = None,
) -> AfterVerdict:
    """Combine every declared upstream condition for one downstream run.

    Args:
        study: The study containing both grids.
        run: The downstream run to judge.
        reports: The known upstream reports, keyed by run path.
        followup: Retry settings used to keep retryable interruptions open.
        now: The current time in UTC Unix seconds, passed by the caller.

    Returns:
        The combined dependency verdict; every ``After`` entry combines with all.
    """
    grid = study.grids[run.grid]
    verdicts = [
        _entry_verdict(
            study=study, run=run, after=after, reports=reports, followup=followup, now=now
        )
        for after in grid.after
    ]
    if AfterVerdict.NEVER in verdicts:
        return AfterVerdict.NEVER
    if all(verdict is AfterVerdict.MET for verdict in verdicts):
        return AfterVerdict.MET
    return AfterVerdict.OPEN


def downstream_of(
    *, study: Study, run_paths: Collection[PurePosixPath], reports: Sequence[RunReport]
) -> list[RunReport]:
    """The waiting, dependency-failed and missing runs that depend on the given runs.

    Args:
        study: The study containing the grids.
        run_paths: The upstream run paths, e.g. the runs selected for resubmission.
        reports: Reports of every run that may depend on them.

    Returns:
        The ``WAITING``, ``DEPENDENCY_FAILED`` and ``MISSING`` reports with a given run among their
        upstream runs, transitively, in the order of ``reports``; the given runs
        themselves are left out.
    """
    given = set(run_paths)
    candidates = [
        report
        for report in reports
        if report.state in (StudyState.WAITING, StudyState.DEPENDENCY_FAILED, StudyState.MISSING)
        and report.run.run_path not in given
    ]
    dependents = _dependents(study=study, candidates=candidates)
    found: set[PurePosixPath] = set()
    queue = deque(given)
    while queue:
        new = [path for path in dependents[queue.popleft()] if path not in found]
        found.update(new)
        queue.extend(new)
    return [report for report in candidates if report.run.run_path in found]


def _dependents(
    *, study: Study, candidates: Sequence[RunReport]
) -> defaultdict[PurePosixPath, list[PurePosixPath]]:
    """The run paths that directly depend on each upstream run path, built once."""
    dependents: defaultdict[PurePosixPath, list[PurePosixPath]] = defaultdict(list)
    for report in candidates:
        for upstream_path in _upstream_paths(study=study, run=report.run):
            dependents[upstream_path].append(report.run.run_path)
    return dependents


def _upstream_paths(*, study: Study, run: StudyRun) -> set[PurePosixPath]:
    """The run paths of every upstream run of ``run``."""
    return {
        upstream.run_path
        for after in study.grids[run.grid].after
        for upstream in upstream_runs(study=study, run=run, after=after)
    }


def _entry_verdict(  # noqa: PLR0913 # why: one input per dependency decision fact
    *,
    study: Study,
    run: StudyRun,
    after: After,
    reports: Mapping[PurePosixPath, RunReport],
    followup: FollowupSettings | None,
    now: float | None,
) -> AfterVerdict:
    """Combine matched runs according to one entry's ``satisfy`` choice."""
    matched = upstream_runs(study=study, run=run, after=after)
    if not matched:
        return AfterVerdict.OPEN
    verdicts = [
        _report_verdict(
            report=reports.get(upstream.run_path),
            condition=after.condition,
            delay_minutes=after.delay_minutes,
            followup=followup,
            now=now,
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
    *,
    report: RunReport | None,
    condition: AfterCondition,
    delay_minutes: int,
    followup: FollowupSettings | None,
    now: float | None,
) -> AfterVerdict:
    """Judge one report, treating an uncollected report as open."""
    if report is None:
        return AfterVerdict.OPEN
    return upstream_verdict(
        condition=condition,
        state=report.state,
        retry_left=retry_left(report=report, followup=followup),
        delay_minutes=delay_minutes,
        started_at=report.job_started_at,
        now=now,
    )


def retry_left(*, report: RunReport, followup: FollowupSettings | None) -> bool:
    """Whether chained retry settings permit another attempt for this report.

    Args:
        report: The upstream run report.
        followup: The retry settings, or ``None`` when follow-up is disabled.

    Returns:
        Whether another attempt remains.
    """
    if followup is None:
        return False
    # why: attempts that ended preempted or node failed do not count, as in the follow-up
    counted = report.attempt - (report.task.uncounted_attempts if report.task else 0)
    return followup.chained_retry is not ChainedRetry.MANUAL and counted < followup.max_attempts


def _ok_verdict(*, facts: _VerdictFacts) -> AfterVerdict:
    """The verdict for an ``OK`` dependency."""
    if facts.state is StudyState.DONE:
        return AfterVerdict.MET
    if facts.state in (StudyState.FAILED, StudyState.DEPENDENCY_FAILED):
        return AfterVerdict.NEVER
    if facts.state is StudyState.INTERRUPTED:
        return AfterVerdict.OPEN if facts.retry_left else AfterVerdict.NEVER
    return AfterVerdict.OPEN


def _any_verdict(*, facts: _VerdictFacts) -> AfterVerdict:
    """The verdict for an ``ANY`` dependency."""
    if facts.state in (StudyState.DONE, StudyState.FAILED, StudyState.DEPENDENCY_FAILED):
        return AfterVerdict.MET
    if facts.state is StudyState.INTERRUPTED:
        return AfterVerdict.OPEN if facts.retry_left else AfterVerdict.MET
    return AfterVerdict.OPEN


def _failed_verdict(*, facts: _VerdictFacts) -> AfterVerdict:
    """The verdict for a ``FAILED`` dependency."""
    if facts.state is StudyState.DONE:
        return AfterVerdict.NEVER
    if facts.state in (StudyState.FAILED, StudyState.DEPENDENCY_FAILED):
        return AfterVerdict.MET
    if facts.state is StudyState.INTERRUPTED:
        return AfterVerdict.OPEN if facts.retry_left else AfterVerdict.MET
    return AfterVerdict.OPEN


def _started_verdict(*, facts: _VerdictFacts) -> AfterVerdict:
    """The verdict for a ``STARTED`` dependency and its elapsed delay."""
    if facts.state is StudyState.DEPENDENCY_FAILED:
        return AfterVerdict.NEVER
    started = facts.state in (
        StudyState.RUNNING,
        StudyState.DONE,
        StudyState.INTERRUPTED,
        StudyState.FAILED,
    )
    if not started:
        return AfterVerdict.OPEN
    if facts.delay_minutes == 0:
        return AfterVerdict.MET
    if facts.started_at is None:
        return AfterVerdict.OPEN
    if facts.now is None:
        return AfterVerdict.OPEN
    delay_seconds = facts.delay_minutes * 60
    return AfterVerdict.MET if facts.now - facts.started_at >= delay_seconds else AfterVerdict.OPEN


_VERDICT_BY_CONDITION: Mapping[AfterCondition, _VerdictStrategy] = MappingProxyType(
    {
        AfterCondition.OK: _ok_verdict,
        AfterCondition.ANY: _any_verdict,
        AfterCondition.FAILED: _failed_verdict,
        AfterCondition.STARTED: _started_verdict,
    }
)

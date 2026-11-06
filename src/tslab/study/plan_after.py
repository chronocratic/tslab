"""Dependency targets for a launch, independent of its launcher."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from tslab.enums import (
    AfterCondition,
    AfterVerdict,
    ChainedRetry,
    LauncherKind,
    Satisfy,
    StudyState,
)
from tslab.study.chain import after_verdict, upstream_verdict
from tslab.study.definition import grid_order, upstream_runs
from tslab.study.records import latest_tasks

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import PurePosixPath

    from tslab.study.definition import After, Study, StudyRun
    from tslab.study.machine.profile import FollowupSettings
    from tslab.study.records import JobRecord, TaskRecord
    from tslab.study.reporting.state import RunReport

__all__ = ['AfterTarget', 'local_after_problems', 'order_runs', 'plan_targets']

_LIVE_STATES = (StudyState.QUEUED, StudyState.WAITING, StudyState.RUNNING)


@dataclass(frozen=True, kw_only=True)
class AfterTarget:
    """The upstream work one downstream job must wait for.

    Args:
        condition: The condition applied to each target.
        delay_minutes: The delay after an upstream starts, for ``STARTED``.
        satisfy: Whether all targets or any target must meet the condition.
        job_tasks: Existing upstream task ids in ``<job id>_<index>`` form.
        run_paths: Upstream runs submitted in the same launch.
    """

    condition: AfterCondition
    delay_minutes: int
    satisfy: Satisfy
    job_tasks: tuple[str, ...]
    run_paths: tuple[PurePosixPath, ...]


@dataclass(frozen=True, kw_only=True)
class _TargetReference:
    """One existing task id or one run path from the current launch."""

    job_task: str | None = None
    run_path: PurePosixPath | None = None


@dataclass(frozen=True, kw_only=True)
class _TargetState:
    """The shared inputs used while resolving all selected runs."""

    study: Study
    reports: Mapping[PurePosixPath, RunReport]
    submitting: set[PurePosixPath]
    latest: Mapping[PurePosixPath, tuple[JobRecord, TaskRecord]]
    followup: FollowupSettings | None


def order_runs(*, study: Study, runs: Sequence[StudyRun]) -> list[StudyRun]:
    """Stably order runs by their grids' dependency order.

    Args:
        study: The study that declares grid dependencies.
        runs: The runs to order.

    Returns:
        Runs grouped in topological grid order, preserving each grid's input order.
    """
    order = {name: index for index, name in enumerate(grid_order(study=study))}
    return sorted(runs, key=lambda run: order[run.grid])


def local_after_problems(
    *, study: Study, runs: Sequence[StudyRun], launcher: LauncherKind
) -> list[str]:
    """The dependency conditions a local launch cannot represent.

    Args:
        study: The study whose grids declare conditions.
        runs: Runs being planned.
        launcher: The machine launcher.

    Returns:
        One message per selected grid that uses an unsupported condition.
    """
    if launcher is not LauncherKind.LOCAL:
        return []
    grids = dict.fromkeys(run.grid for run in runs)
    return [
        f'grid {grid_name!r}: condition {after.condition.value!r} needs {LauncherKind.SLURM.name}'
        for grid_name in grids
        for after in study.grids[grid_name].after
        if after.condition is AfterCondition.STARTED
    ]


def plan_targets(  # noqa: PLR0913 # why: public interface names each planning input
    *,
    study: Study,
    runs: Sequence[StudyRun],
    reports: Mapping[PurePosixPath, RunReport],
    submitting: frozenset[PurePosixPath],
    jobs: Sequence[JobRecord],
    followup: FollowupSettings | None = None,
) -> tuple[dict[PurePosixPath, tuple[AfterTarget, ...]], frozenset[PurePosixPath]]:
    """Resolve upstream reports into existing-job or same-launch targets.

    Args:
        study: The study whose grids declare dependencies.
        runs: The selected runs considered for this launch.
        reports: Reports for selected and upstream runs, keyed by run path.
        submitting: Runs that would enter this launch before dependency filtering.
        jobs: Recorded submissions used to resolve live tasks and array indices.
        followup: Retry settings used to keep eligible interruptions open.

    Returns:
        Targets per downstream run and the downstream runs that cannot be submitted.
    """
    pending = set(submitting)
    state = _TargetState(
        study=study,
        reports=reports,
        submitting=pending,
        latest=latest_tasks(jobs=jobs),
        followup=followup,
    )
    targets: dict[PurePosixPath, tuple[AfterTarget, ...]] = {}
    blocked: set[PurePosixPath] = set()
    for run in order_runs(study=study, runs=runs):
        path = run.run_path
        run_targets, is_blocked = _run_targets(state=state, run=run)
        if is_blocked:
            blocked.add(path)
            pending.discard(path)
            continue
        targets[path] = run_targets
    return targets, frozenset(blocked)


def _run_targets(*, state: _TargetState, run: StudyRun) -> tuple[tuple[AfterTarget, ...], bool]:
    """Resolve one run's conditions and invalidate targets for impossible work."""
    grid = state.study.grids[run.grid]
    if not grid.after:
        return (), False
    verdict_reports = _reports_for_launch(reports=state.reports, submitting=state.submitting)
    verdict = after_verdict(
        study=state.study, run=run, reports=verdict_reports, followup=state.followup
    )
    if verdict is AfterVerdict.NEVER:
        return (), True
    entries = [_entry_target(state=state, run=run, after=after) for after in grid.after]
    if any(is_blocked for _, is_blocked in entries):
        return (), True
    return tuple(target for target, _ in entries if target is not None), False


def _reports_for_launch(
    *, reports: Mapping[PurePosixPath, RunReport], submitting: set[PurePosixPath]
) -> dict[PurePosixPath, RunReport]:
    """Treat selected upstream runs as open because this launch starts a new attempt."""
    return {
        path: replace(report, state=StudyState.MISSING) if path in submitting else report
        for path, report in reports.items()
    }


def _entry_target(
    *, state: _TargetState, run: StudyRun, after: After
) -> tuple[AfterTarget | None, bool]:
    """Build one entry's target, enforcing its all/any rule."""
    upstream = upstream_runs(study=state.study, run=run, after=after)
    if not upstream:
        return None, True
    verdicts = [
        _run_verdict(run=upstream_run, condition=after.condition, state=state)
        for upstream_run in upstream
    ]
    return _satisfy_target(upstream=upstream, verdicts=verdicts, after=after, state=state)


def _run_verdict(*, state: _TargetState, run: StudyRun, condition: AfterCondition) -> AfterVerdict:
    """Judge one upstream, with an in-launch rerun treated as open."""
    if run.run_path in state.submitting:
        return AfterVerdict.OPEN
    report = state.reports.get(run.run_path)
    if report is None:
        return AfterVerdict.OPEN
    return upstream_verdict(
        condition=condition,
        state=report.state,
        retry_left=_retry_left(report=report, followup=state.followup),
    )


def _satisfy_target(
    *,
    state: _TargetState,
    upstream: Sequence[StudyRun],
    verdicts: Sequence[AfterVerdict],
    after: After,
) -> tuple[AfterTarget | None, bool]:
    """Choose which open targets make one dependency entry safe to schedule."""
    if after.satisfy is Satisfy.ALL and AfterVerdict.NEVER in verdicts:
        return None, True
    if after.satisfy is Satisfy.ANY and AfterVerdict.MET in verdicts:
        return None, False
    open_runs = [
        run for run, verdict in zip(upstream, verdicts, strict=True) if verdict is AfterVerdict.OPEN
    ]
    if not open_runs:
        return None, AfterVerdict.NEVER in verdicts
    references = [_target_reference(state=state, run=run) for run in open_runs]
    if after.satisfy is Satisfy.ALL and any(reference is None for reference in references):
        return None, True
    usable = [reference for reference in references if reference is not None]
    if not usable:
        return None, True
    job_tasks = tuple(dict.fromkeys(ref.job_task for ref in usable if ref.job_task is not None))
    run_paths = tuple(dict.fromkeys(ref.run_path for ref in usable if ref.run_path is not None))
    return (
        AfterTarget(
            condition=after.condition,
            delay_minutes=after.delay_minutes,
            satisfy=after.satisfy,
            job_tasks=job_tasks,
            run_paths=run_paths,
        ),
        False,
    )


def _target_reference(*, state: _TargetState, run: StudyRun) -> _TargetReference | None:
    """Return the scheduled path or live job task for one open upstream."""
    if run.run_path in state.submitting:
        return _TargetReference(run_path=run.run_path)
    report = state.reports.get(run.run_path)
    if report is None or report.state not in _LIVE_STATES:
        return None
    record = state.latest.get(run.run_path)
    if record is None:
        return None
    job, task = record
    return _TargetReference(job_task=f'{job.job_id}_{task.index}')


def _retry_left(*, report: RunReport, followup: FollowupSettings | None) -> bool:
    """Whether configured chained retries leave another attempt for a report."""
    if followup is None:
        return False
    return (
        followup.chained_retry is not ChainedRetry.MANUAL and report.attempt < followup.max_attempts
    )

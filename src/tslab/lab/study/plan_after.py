"""Dependency targets for a launch, independent of its launcher."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING

from tslab.enums import (
    AfterCondition,
    AfterVerdict,
    ChainedRetry,
    LauncherKind,
    Satisfy,
    StudyState,
)
from tslab.study.chain import after_verdict, retry_left, upstream_verdict
from tslab.study.definition import grid_order, match_axes, upstream_runs
from tslab.study.records import latest_tasks

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import PurePosixPath

    from tslab.study.definition import After, Study, StudyRun
    from tslab.study.machine.profile import FollowupSettings
    from tslab.study.plan import JobGroup
    from tslab.study.records import JobRecord, TaskRecord
    from tslab.study.reporting.state import RunReport

__all__ = ['AfterTarget', 'local_after_problems', 'merge_corr_groups', 'order_runs', 'plan_targets']

_LIVE_STATES = (StudyState.QUEUED, StudyState.WAITING, StudyState.RUNNING)


@dataclass(frozen=True, kw_only=True)
class AfterTarget:
    """The upstream work one downstream job must wait for.

    Attributes:
        condition: The condition applied to each target.
        delay_minutes: The delay after an upstream starts, for ``STARTED``.
        satisfy: Whether all targets or any target must meet the condition.
        job_tasks: Existing upstream task ids in ``<job id>_<index>`` form.
        run_paths: Upstream runs submitted in the same launch.
        corr_ok: The run pairs one to one with a single upstream run, so a group of such
            runs may wait with ``aftercorr`` when the array indices line up.
        corr: The target waits for the matching task of one upstream array; ``job_tasks``
            holds the array job id, or ``run_paths`` one run of the upstream array.
    """

    condition: AfterCondition
    delay_minutes: int
    satisfy: Satisfy
    job_tasks: tuple[str, ...]
    run_paths: tuple[PurePosixPath, ...]
    corr_ok: bool = False
    corr: bool = False


@dataclass(frozen=True, kw_only=True)
class _TargetReference:
    """One existing task id or one run path from the current launch.

    Attributes:
        job_task: The existing SLURM job task id, when available.
        run_path: The run path submitted in this launch, when available.
    """

    job_task: str | None = None
    run_path: PurePosixPath | None = None


@dataclass(frozen=True, kw_only=True)
class _TargetState:
    """The shared inputs used while resolving all selected runs.

    Attributes:
        study: The study whose grid dependencies are being planned.
        reports: The known state of selected and upstream runs.
        latest: The latest recorded job and task for each run.
        followup: Retry settings used to keep retryable runs open.
        defer_failed_targets: Whether SLURM retries hold failed dependencies open.
        now: The current time in UTC Unix seconds, when known.
    """

    study: Study
    reports: Mapping[PurePosixPath, RunReport]
    latest: Mapping[PurePosixPath, tuple[JobRecord, TaskRecord]]
    followup: FollowupSettings | None
    defer_failed_targets: bool
    now: float | None

    def __post_init__(self) -> None:
        """Copy collection inputs so the planning snapshot cannot change."""
        object.__setattr__(self, 'reports', MappingProxyType(dict(self.reports)))
        object.__setattr__(self, 'latest', MappingProxyType(dict(self.latest)))


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
    launcher: LauncherKind | None = None,
    now: float | None = None,
) -> tuple[dict[PurePosixPath, tuple[AfterTarget, ...]], frozenset[PurePosixPath]]:
    """Resolve upstream reports into existing-job or same-launch targets.

    Args:
        study: The study whose grids declare dependencies.
        runs: The selected runs considered for this launch.
        reports: Reports for selected and upstream runs, keyed by run path.
        submitting: Runs that would enter this launch before dependency filtering.
        jobs: Recorded submissions used to resolve live tasks and array indices.
        followup: Retry settings used to keep eligible interruptions open.
        launcher: The launcher that will represent open dependencies.
        now: The current time in UTC Unix seconds, passed by the caller.

    Returns:
        Targets per downstream run and the downstream runs that cannot be submitted.
    """
    pending = set(submitting)
    state = _TargetState(
        study=study,
        reports=reports,
        latest=latest_tasks(jobs=jobs),
        followup=followup,
        defer_failed_targets=launcher is LauncherKind.SLURM,
        now=now,
    )
    targets: dict[PurePosixPath, tuple[AfterTarget, ...]] = {}
    blocked: set[PurePosixPath] = set()
    for run in order_runs(study=study, runs=runs):
        path = run.run_path
        run_targets, is_blocked = _run_targets(state=state, run=run, submitting=frozenset(pending))
        if is_blocked:
            blocked.add(path)
            pending.discard(path)
            continue
        targets[path] = run_targets
    return targets, frozenset(blocked)


def _run_targets(
    *, state: _TargetState, run: StudyRun, submitting: frozenset[PurePosixPath]
) -> tuple[tuple[AfterTarget, ...], bool]:
    """Resolve one run's conditions and invalidate targets for impossible work."""
    grid = state.study.grids[run.grid]
    if not grid.after:
        return (), False
    verdict_reports = _reports_for_launch(reports=state.reports, submitting=submitting)
    verdict = after_verdict(
        study=state.study, run=run, reports=verdict_reports, followup=state.followup, now=state.now
    )
    if verdict is AfterVerdict.NEVER:
        return (), True
    entries = [
        _entry_target(state=state, run=run, after=after, submitting=submitting)
        for after in grid.after
    ]
    if any(is_blocked for _, is_blocked in entries):
        return (), True
    found = tuple(target for target, _ in entries if target is not None)
    return _mark_corr_candidate(state=state, run=run, targets=found), False


def _mark_corr_candidate(
    *, state: _TargetState, run: StudyRun, targets: tuple[AfterTarget, ...]
) -> tuple[AfterTarget, ...]:
    """Flag the target of a run that pairs one to one with one upstream run."""
    grid = state.study.grids[run.grid]
    if len(grid.after) != 1 or len(targets) != 1:
        return targets
    (after,) = grid.after
    (target,) = targets
    paired = set(match_axes(study=state.study, grid_name=run.grid, after=after))
    single = len(upstream_runs(study=state.study, run=run, after=after)) == 1
    one_id = len(target.job_tasks) + len(target.run_paths) == 1
    plain = after.condition is AfterCondition.OK and after.satisfy is Satisfy.ALL
    if single and one_id and plain and set(grid.axes) <= paired:
        return (replace(target, corr_ok=True),)
    return targets


def _reports_for_launch(
    *, reports: Mapping[PurePosixPath, RunReport], submitting: frozenset[PurePosixPath]
) -> dict[PurePosixPath, RunReport]:
    """Treat selected upstream runs as open because this launch starts a new attempt."""
    return {
        path: replace(report, state=StudyState.MISSING) if path in submitting else report
        for path, report in reports.items()
    }


def _entry_target(
    *, state: _TargetState, run: StudyRun, after: After, submitting: frozenset[PurePosixPath]
) -> tuple[AfterTarget | None, bool]:
    """Build one entry's target, enforcing its all/any rule."""
    upstream = upstream_runs(study=state.study, run=run, after=after)
    if not upstream:
        return None, True
    verdicts = [
        _run_verdict(
            run=upstream_run,
            condition=after.condition,
            delay_minutes=after.delay_minutes,
            state=state,
            submitting=submitting,
        )
        for upstream_run in upstream
    ]
    return _satisfy_target(
        upstream=upstream, verdicts=verdicts, after=after, state=state, submitting=submitting
    )


def _run_verdict(
    *,
    state: _TargetState,
    run: StudyRun,
    condition: AfterCondition,
    delay_minutes: int,
    submitting: frozenset[PurePosixPath],
) -> AfterVerdict:
    """Judge one upstream, with an in-launch rerun treated as open."""
    if run.run_path in submitting:
        return AfterVerdict.OPEN
    report = state.reports.get(run.run_path)
    if report is None:
        return AfterVerdict.OPEN
    return upstream_verdict(
        condition=condition,
        state=report.state,
        retry_left=retry_left(report=report, followup=state.followup),
        delay_minutes=delay_minutes,
        started_at=report.job_started_at,
        now=state.now,
    )


def _satisfy_target(
    *,
    state: _TargetState,
    upstream: Sequence[StudyRun],
    verdicts: Sequence[AfterVerdict],
    after: After,
    submitting: frozenset[PurePosixPath],
) -> tuple[AfterTarget | None, bool]:
    """Choose which open targets make one dependency entry safe to schedule."""
    if after.satisfy is Satisfy.ANY and AfterVerdict.MET in verdicts:
        return None, False
    if (after.satisfy is Satisfy.ALL and AfterVerdict.NEVER in verdicts) or _defer_failed_target(
        after=after, verdicts=verdicts, state=state
    ):
        return None, True
    open_runs = [
        run for run, verdict in zip(upstream, verdicts, strict=True) if verdict is AfterVerdict.OPEN
    ]
    if not open_runs:
        return None, AfterVerdict.NEVER in verdicts
    references = [
        _target_reference(state=state, run=run, submitting=submitting) for run in open_runs
    ]
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


def _defer_failed_target(
    *, after: After, verdicts: Sequence[AfterVerdict], state: _TargetState
) -> bool:
    """Hold SLURM dependents until retryable upstream attempts reach a final state."""
    return (
        after.condition is AfterCondition.FAILED
        and state.defer_failed_targets
        and state.followup is not None
        and state.followup.chained_retry is not ChainedRetry.MANUAL
        and AfterVerdict.OPEN in verdicts
    )


def _target_reference(
    *, state: _TargetState, run: StudyRun, submitting: frozenset[PurePosixPath]
) -> _TargetReference | None:
    """Return the scheduled path or live job task for one open upstream."""
    if run.run_path in submitting:
        return _TargetReference(run_path=run.run_path)
    report = state.reports.get(run.run_path)
    if report is None or report.state not in _LIVE_STATES:
        return None
    record = state.latest.get(run.run_path)
    if record is None:
        return None
    job, task = record
    return _TargetReference(job_task=f'{job.job_id}_{task.index}')


def merge_corr_groups(
    *, groups: Sequence[JobGroup], indices: Mapping[PurePosixPath, int]
) -> list[JobGroup]:
    """Merge one-to-one downstream groups into arrays that wait with ``aftercorr``.

    A group qualifies when its only target is a one-to-one candidate and every task
    has the array index of its upstream run. Qualifying groups of one grid, job name,
    resources and code snapshot that wait on the same upstream array become one group.

    Args:
        groups: The planned groups, numbered from the first job file.
        indices: The run path to array index of the launch.

    Returns:
        The groups with the merged ones in the place of the first, renumbered in order.
    """
    owner = {
        task.resolved.run.run_path: position
        for position, group in enumerate(groups)
        for task in group.tasks
    }
    merged: dict[tuple[object, ...], int] = {}
    result: list[JobGroup] = []
    for group in groups:
        key = _corr_key(group=group, owner=owner, indices=indices)
        if key is None:
            result.append(group)
        elif key in merged:
            position = merged[key]
            tasks = (*result[position].tasks, *group.tasks)
            result[position] = replace(
                result[position], tasks=tuple(sorted(tasks, key=lambda task: task.index))
            )
        else:
            merged[key] = len(result)
            result.append(
                replace(group, after=(_corr_target(group=group, groups=groups, owner=owner),))
            )
    first = groups[0].job_file if groups else 0
    return [replace(group, job_file=first + position) for position, group in enumerate(result)]


def _upstream_ref(
    *, target: AfterTarget, owner: Mapping[PurePosixPath, int], indices: Mapping[PurePosixPath, int]
) -> tuple[object, int] | None:
    """The upstream array and the index of the upstream task a target waits for."""
    if target.job_tasks:
        array_id, _, index = target.job_tasks[0].rpartition('_')
        return ('job', array_id), int(index)
    path = target.run_paths[0]
    if path not in owner or path not in indices:
        return None
    return ('group', owner[path]), indices[path]


def _corr_key(
    *, group: JobGroup, owner: Mapping[PurePosixPath, int], indices: Mapping[PurePosixPath, int]
) -> tuple[object, ...] | None:
    """The merge key of a group that can wait with ``aftercorr``; ``None`` if it cannot."""
    if len(group.after) != 1 or not group.after[0].corr_ok or group.after[0].corr:
        return None
    reference = _upstream_ref(target=group.after[0], owner=owner, indices=indices)
    if reference is None:
        return None
    array, upstream_index = reference
    if any(task.index != upstream_index for task in group.tasks):
        return None
    return (group.grid, group.job_name, group.resources, group.code_snapshot, array)


def _corr_target(
    *, group: JobGroup, groups: Sequence[JobGroup], owner: Mapping[PurePosixPath, int]
) -> AfterTarget:
    """The ``aftercorr`` target that replaces the one-to-one target of a group."""
    (target,) = group.after
    if target.job_tasks:
        array_id = target.job_tasks[0].rpartition('_')[0]
        return replace(target, job_tasks=(array_id,), run_paths=(), corr=True)
    upstream = groups[owner[target.run_paths[0]]]
    return replace(target, run_paths=(upstream.tasks[0].resolved.run.run_path,), corr=True)

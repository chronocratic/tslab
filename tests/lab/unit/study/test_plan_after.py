"""Dependency targets, run ordering and target-aware job grouping."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import TYPE_CHECKING

import pytest

from tests.support.fake_backend import FakeResources
from tslab.core.status import utc_now_iso
from tslab.enums import AfterCondition, ChainedRetry, JobPhase, LauncherKind, Satisfy, StudyState
from tslab.study import After, Grid, Study
from tslab.study.definition import expand_runs, StudyRun
from tslab.study.launchers.slurm.dependency import dependency_text
from tslab.study.launchers.slurm.records import SlurmJobRecord
from tslab.study.machine import MachineProfile
from tslab.study.machine.profile import FollowupSettings
from tslab.study.plan import assign_indices, JobGroup, plan_jobs, ResolvedRun
from tslab.study.plan_after import (
    _TargetState,
    AfterTarget,
    local_after_problems,
    order_runs,
    plan_targets,
)
from tslab.study.records import JobKind, ProgressSnapshot, TaskRecord
from tslab.study.reporting.state import FolderFacts, RunReport

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


@dataclass(frozen=True, kw_only=True)
class _ChainConfig:
    """The config fields needed by study expansion and argument planning."""

    seed: int = 0
    output_root: Path = Path('runs')
    run_path: Path = Path('run')


def _study(
    *, fan_out: bool = False, condition: AfterCondition = AfterCondition.OK, delay_minutes: int = 0
) -> Study:
    upstream = Grid(
        short_name='pre',
        command=('python', '-m', 'pretrain'),
        config=_ChainConfig,
        axes={'dataset': {'a': {}, 'b': {}}},
        run_path='{grid}/{dataset}',
    )
    downstream = Grid(
        short_name='probe',
        command=('python', '-m', 'probe'),
        config=_ChainConfig,
        axes={'dataset': {'a': {}, 'b': {}}, 'head': {'svm': {}, 'knn': {}}},
        after=(
            After(
                grid='pretrain',
                match=() if fan_out else None,
                condition=condition,
                delay_minutes=delay_minutes,
            ),
        ),
    )
    return Study(name='chain', grids={'probe': downstream, 'pretrain': upstream})


def _report(
    *,
    run: StudyRun,
    state: StudyState,
    job_id: str | None = None,
    job_started_at: float | None = None,
) -> RunReport:
    phase = {StudyState.QUEUED: JobPhase.QUEUED, StudyState.RUNNING: JobPhase.RUNNING}.get(
        state, JobPhase.UNKNOWN
    )
    return RunReport(
        run=run,
        state=state,
        reason=None,
        job_id=job_id,
        job_state=None,
        job_phase=phase,
        attempt=1,
        task=None,
        folder=FolderFacts(snapshot=None, read_error=None, lock_live=False),
        job_started_at=job_started_at,
    )


def _job_record(*, job_id: str, runs: Sequence[StudyRun], first_index: int = 0) -> SlurmJobRecord:
    tasks = tuple(
        TaskRecord(
            index=first_index + position,
            grid=run.grid,
            run_path=run.run_path,
            command=(),
            extra_args=(),
            resources={},
            attempt=1,
            progress=ProgressSnapshot(done_instances=0, newest_checkpoint_mtime=None),
        )
        for position, run in enumerate(runs)
    )
    return SlurmJobRecord(
        kind=JobKind.RUN,
        submission_id='submit-1',
        time=utc_now_iso(),
        launcher=LauncherKind.SLURM,
        job_id=job_id,
        job_file=1,
        tasks=tasks,
    )


def _plan_groups(
    *,
    study: Study,
    runs: Sequence[StudyRun],
    after: Mapping[PurePosixPath, tuple[AfterTarget, ...]],
    tmp_path: Path,
    indices: Mapping[PurePosixPath, int] | None = None,
) -> list[JobGroup]:
    machine = MachineProfile(
        path=tmp_path / 'machine.toml', launcher=LauncherKind.LOCAL, output_root=tmp_path / 'runs'
    )
    resolved = [
        ResolvedRun(run=run, settings=run.settings, resources=FakeResources()) for run in runs
    ]
    return plan_jobs(
        study=study,
        machine=machine,
        resolved=resolved,
        indices=indices or {run.run_path: position for position, run in enumerate(runs)},
        extra_args={},
        attempts={},
        first_job_file=1,
        environment={},
        after=after,
    )


def test_done_upstream_gives_no_target() -> None:
    study = _study()
    runs = expand_runs(study=study)
    downstream = next(run for run in runs if run.grid == 'probe')
    upstream = next(run for run in runs if run.run_path == PurePosixPath('pretrain/a'))

    targets, blocked = plan_targets(
        study=study,
        runs=[downstream],
        reports={upstream.run_path: _report(run=upstream, state=StudyState.DONE)},
        submitting=frozenset(),
        jobs=(),
    )

    assert targets[downstream.run_path] == ()
    assert blocked == frozenset()


def test_live_upstream_gives_job_task() -> None:
    study = _study()
    runs = expand_runs(study=study)
    downstream = next(run for run in runs if run.grid == 'probe')
    upstream = next(run for run in runs if run.run_path == PurePosixPath('pretrain/a'))
    jobs = [_job_record(job_id='41', runs=[upstream], first_index=3)]

    targets, _ = plan_targets(
        study=study,
        runs=[downstream],
        reports={upstream.run_path: _report(run=upstream, state=StudyState.RUNNING)},
        submitting=frozenset(),
        jobs=jobs,
    )

    assert targets[downstream.run_path] == (
        AfterTarget(
            condition=AfterCondition.OK,
            delay_minutes=0,
            satisfy=Satisfy.ALL,
            job_tasks=('41_3',),
            run_paths=(),
        ),
    )


def test_target_state_copies_and_freezes_mapping_inputs() -> None:
    study = _study()
    upstream = next(run for run in expand_runs(study=study) if run.grid == 'pretrain')
    report = _report(run=upstream, state=StudyState.RUNNING)
    job = _job_record(job_id='41', runs=[upstream])
    reports = {upstream.run_path: report}
    latest = {upstream.run_path: (job, job.tasks[0])}

    state = _TargetState(
        study=study,
        reports=reports,
        latest=latest,
        followup=None,
        defer_failed_targets=False,
        now=None,
    )
    reports.clear()
    latest.clear()

    assert state.reports[upstream.run_path] == report
    assert state.latest[upstream.run_path] == (job, job.tasks[0])
    assert isinstance(state.reports, MappingProxyType)
    assert isinstance(state.latest, MappingProxyType)


@pytest.mark.parametrize(
    ('elapsed_minutes', 'expected_dependency'), [(1, 'after:41_0+30'), (60, None)]
)
def test_started_dependency_waits_from_upstream_start_time(
    elapsed_minutes: int, expected_dependency: str | None, tmp_path: Path
) -> None:
    study = _study(condition=AfterCondition.STARTED, delay_minutes=30)
    runs = expand_runs(study=study)
    downstream = next(run for run in runs if run.grid == 'probe')
    upstream = next(run for run in runs if run.run_path == PurePosixPath('pretrain/a'))

    targets, blocked = plan_targets(
        study=study,
        runs=[downstream],
        reports={
            upstream.run_path: _report(
                run=upstream, state=StudyState.RUNNING, job_id='41_0', job_started_at=1_000.0
            )
        },
        submitting=frozenset(),
        jobs=[_job_record(job_id='41', runs=[upstream])],
        launcher=LauncherKind.SLURM,
        now=1_000.0 + elapsed_minutes * 60,
    )

    expected_target = (
        (
            AfterTarget(
                condition=AfterCondition.STARTED,
                delay_minutes=30,
                satisfy=Satisfy.ALL,
                job_tasks=('41_0',),
                run_paths=(),
            ),
        )
        if expected_dependency is not None
        else ()
    )
    assert targets[downstream.run_path] == expected_target
    if expected_dependency is not None:
        assert dependency_text(after=expected_target, upstream={}) == expected_dependency
    assert blocked == frozenset()


def test_started_dependency_with_zero_delay_needs_no_target() -> None:
    study = _study(condition=AfterCondition.STARTED)
    runs = expand_runs(study=study)
    downstream = next(run for run in runs if run.grid == 'probe')
    upstream = next(run for run in runs if run.run_path == PurePosixPath('pretrain/a'))

    targets, blocked = plan_targets(
        study=study,
        runs=[downstream],
        reports={
            upstream.run_path: _report(
                run=upstream, state=StudyState.RUNNING, job_started_at=1_000.0
            )
        },
        submitting=frozenset(),
        jobs=[_job_record(job_id='41', runs=[upstream])],
        launcher=LauncherKind.SLURM,
        now=1_000.0,
    )

    assert targets[downstream.run_path] == ()
    assert blocked == frozenset()


def test_started_dependency_on_finished_upstream_after_delay_needs_no_target() -> None:
    study = _study(condition=AfterCondition.STARTED, delay_minutes=30)
    runs = expand_runs(study=study)
    downstream = next(run for run in runs if run.grid == 'probe')
    upstream = next(run for run in runs if run.run_path == PurePosixPath('pretrain/a'))

    targets, blocked = plan_targets(
        study=study,
        runs=[downstream],
        reports={
            upstream.run_path: _report(run=upstream, state=StudyState.DONE, job_started_at=1_000.0)
        },
        submitting=frozenset(),
        jobs=(),
        launcher=LauncherKind.SLURM,
        now=1_000.0 + 60 * 60,
    )

    assert targets[downstream.run_path] == ()
    assert blocked == frozenset()


def test_local_machine_rejects_started_dependency() -> None:
    study = _study(condition=AfterCondition.STARTED, delay_minutes=30)
    runs = expand_runs(study=study)

    problems = local_after_problems(study=study, runs=runs, launcher=LauncherKind.LOCAL)

    assert problems == ["grid 'probe': condition 'started' needs SLURM"]


def test_upstream_in_same_submit_gives_run_path() -> None:
    study = _study()
    runs = expand_runs(study=study)
    downstream = next(run for run in runs if run.grid == 'probe')
    upstream = next(run for run in runs if run.run_path == PurePosixPath('pretrain/a'))

    targets, _ = plan_targets(
        study=study,
        runs=[downstream],
        reports={upstream.run_path: _report(run=upstream, state=StudyState.MISSING)},
        submitting=frozenset({upstream.run_path, downstream.run_path}),
        jobs=(),
    )

    assert targets[downstream.run_path][0].run_paths == (upstream.run_path,)
    assert targets[downstream.run_path][0].job_tasks == ()


def test_failed_upstream_blocks_downstream() -> None:
    study = _study()
    runs = expand_runs(study=study)
    downstream = next(run for run in runs if run.grid == 'probe')
    upstream = next(run for run in runs if run.run_path == PurePosixPath('pretrain/a'))

    _, blocked = plan_targets(
        study=study,
        runs=[downstream],
        reports={upstream.run_path: _report(run=upstream, state=StudyState.FAILED)},
        submitting=frozenset(),
        jobs=(),
    )

    assert blocked == frozenset({downstream.run_path})


@pytest.mark.parametrize('retry', [ChainedRetry.RESUBMIT, ChainedRetry.REQUEUE])
def test_failed_condition_waits_for_retryable_slurm_upstream(
    tmp_path: Path, retry: ChainedRetry
) -> None:
    study = _study(condition=AfterCondition.FAILED)
    runs = expand_runs(study=study)
    downstream = next(run for run in runs if run.grid == 'probe')
    upstream = next(run for run in runs if run.run_path == PurePosixPath('pretrain/a'))
    machine = MachineProfile(
        path=tmp_path / 'machine.toml', launcher=LauncherKind.SLURM, output_root=tmp_path / 'runs'
    )

    targets, blocked = plan_targets(
        study=study,
        runs=[downstream],
        reports={upstream.run_path: _report(run=upstream, state=StudyState.RUNNING, job_id='41_0')},
        submitting=frozenset(),
        jobs=[_job_record(job_id='41', runs=[upstream])],
        followup=FollowupSettings(chained_retry=retry),
        launcher=machine.launcher,
    )

    assert downstream.run_path not in targets
    assert blocked == frozenset({downstream.run_path})


def test_failed_condition_accepts_terminal_slurm_failure(tmp_path: Path) -> None:
    study = _study(condition=AfterCondition.FAILED)
    runs = expand_runs(study=study)
    downstream = next(run for run in runs if run.grid == 'probe')
    upstream = next(run for run in runs if run.run_path == PurePosixPath('pretrain/a'))
    machine = MachineProfile(
        path=tmp_path / 'machine.toml', launcher=LauncherKind.SLURM, output_root=tmp_path / 'runs'
    )

    targets, blocked = plan_targets(
        study=study,
        runs=[downstream],
        reports={upstream.run_path: _report(run=upstream, state=StudyState.FAILED)},
        submitting=frozenset(),
        jobs=(),
        followup=FollowupSettings(chained_retry=ChainedRetry.RESUBMIT),
        launcher=machine.launcher,
    )

    assert targets[downstream.run_path] == ()
    assert blocked == frozenset()


@pytest.mark.parametrize('retry', [ChainedRetry.RESUBMIT, ChainedRetry.REQUEUE])
def test_failed_condition_defers_same_launch_slurm_dependent(
    tmp_path: Path, retry: ChainedRetry
) -> None:
    study = _study(condition=AfterCondition.FAILED)
    runs = expand_runs(study=study)
    downstream = next(run for run in runs if run.grid == 'probe')
    upstream = next(run for run in runs if run.run_path == PurePosixPath('pretrain/a'))

    targets, blocked = plan_targets(
        study=study,
        runs=[upstream, downstream],
        reports={upstream.run_path: _report(run=upstream, state=StudyState.MISSING)},
        submitting=frozenset({upstream.run_path, downstream.run_path}),
        jobs=(),
        followup=FollowupSettings(chained_retry=retry),
        launcher=LauncherKind.SLURM,
    )

    assert downstream.run_path not in targets
    assert blocked == frozenset({downstream.run_path})


@pytest.mark.parametrize('retry', [ChainedRetry.MANUAL])
def test_failed_condition_keeps_native_dependency_without_resubmit_retry(
    tmp_path: Path, retry: ChainedRetry
) -> None:
    study = _study(condition=AfterCondition.FAILED)
    runs = expand_runs(study=study)
    downstream = next(run for run in runs if run.grid == 'probe')
    upstream = next(run for run in runs if run.run_path == PurePosixPath('pretrain/a'))

    targets, blocked = plan_targets(
        study=study,
        runs=[downstream],
        reports={upstream.run_path: _report(run=upstream, state=StudyState.RUNNING)},
        submitting=frozenset(),
        jobs=[_job_record(job_id='41', runs=[upstream])],
        followup=FollowupSettings(chained_retry=retry),
        launcher=LauncherKind.SLURM,
    )

    assert targets[downstream.run_path][0].job_tasks == ('41_0',)
    assert dependency_text(after=targets[downstream.run_path], upstream={}) == 'afternotok:41_0'
    assert blocked == frozenset()


def test_unselected_missing_upstream_blocks_downstream() -> None:
    study = _study()
    runs = expand_runs(study=study)
    downstream = next(run for run in runs if run.grid == 'probe')
    upstream = next(run for run in runs if run.run_path == PurePosixPath('pretrain/a'))

    _, blocked = plan_targets(
        study=study,
        runs=[downstream],
        reports={upstream.run_path: _report(run=upstream, state=StudyState.MISSING)},
        submitting=frozenset(),
        jobs=(),
    )

    assert blocked == frozenset({downstream.run_path})


def test_fan_out_groups_share_one_array(tmp_path: Path) -> None:
    study = _study(fan_out=True)
    runs = expand_runs(study=study)
    upstream = [run for run in runs if run.grid == 'pretrain']
    downstream = [run for run in runs if run.grid == 'probe']
    jobs = [_job_record(job_id='41', runs=upstream)]
    reports = {run.run_path: _report(run=run, state=StudyState.RUNNING) for run in upstream}
    targets, _ = plan_targets(
        study=study, runs=downstream, reports=reports, submitting=frozenset(), jobs=jobs
    )

    groups = _plan_groups(study=study, runs=downstream, after=targets, tmp_path=tmp_path)

    assert len(groups) == 1


def test_matched_runs_split_by_target(tmp_path: Path) -> None:
    study = _study()
    runs = expand_runs(study=study)
    upstream = [run for run in runs if run.grid == 'pretrain']
    downstream = [run for run in runs if run.grid == 'probe']
    jobs = [_job_record(job_id='41', runs=upstream)]
    reports = {run.run_path: _report(run=run, state=StudyState.RUNNING) for run in upstream}
    targets, _ = plan_targets(
        study=study, runs=downstream, reports=reports, submitting=frozenset(), jobs=jobs
    )

    groups = _plan_groups(study=study, runs=downstream, after=targets, tmp_path=tmp_path)

    assert len(groups) == 2
    assert [group.after[0].job_tasks for group in groups] == [('41_0',), ('41_1',)]


def test_order_runs_sorts_grids_stably() -> None:
    study = _study()
    runs = expand_runs(study=study)

    ordered = order_runs(study=study, runs=runs)

    assert [run.grid for run in ordered] == [
        'pretrain',
        'pretrain',
        'probe',
        'probe',
        'probe',
        'probe',
    ]


def test_plan_marks_upstream_and_downstream_groups_chained(tmp_path: Path) -> None:
    study = _study()
    runs = expand_runs(study=study)
    groups = _plan_groups(study=study, runs=runs, after={}, tmp_path=tmp_path)

    assert {group.grid for group in groups if group.chained} == {'pretrain', 'probe'}


# ---- aftercorr -----------------------------------------------------------------


def _one_to_one_study() -> Study:
    upstream = Grid(
        short_name='pre',
        command=('python', '-m', 'pretrain'),
        config=_ChainConfig,
        axes={'dataset': {'a': {}, 'b': {}}},
        run_path='{grid}/{dataset}',
    )
    downstream = Grid(
        short_name='probe',
        command=('python', '-m', 'probe'),
        config=_ChainConfig,
        axes={'dataset': {'a': {}, 'b': {}}},
        run_path='{grid}/{dataset}',
        after=(After(grid='pretrain'),),
    )
    return Study(name='chain', grids={'pretrain': upstream, 'probe': downstream})


def _by_path(*, study: Study) -> dict[str, StudyRun]:
    return {run.run_path.as_posix(): run for run in expand_runs(study=study)}


def test_one_to_one_downstream_takes_upstream_index() -> None:
    study = _one_to_one_study()
    runs = _by_path(study=study)
    jobs = [_job_record(job_id='41', runs=[runs['pretrain/a']], first_index=5)]
    indices = assign_indices(study=study, runs=list(runs.values()), jobs=jobs)
    assert indices[PurePosixPath('pretrain/a')] == 5
    assert indices[PurePosixPath('probe/a')] == 5
    assert indices[PurePosixPath('probe/b')] == indices[PurePosixPath('pretrain/b')]


def test_fan_out_axis_keeps_existing_index_rule() -> None:
    study = _study()
    runs = _by_path(study=study)
    jobs = [_job_record(job_id='41', runs=[runs['pretrain/a']], first_index=5)]
    indices = assign_indices(study=study, runs=list(runs.values()), jobs=jobs)
    probes = [indices[path] for path in indices if path.parts[0] == 'probe']
    assert sorted(probes) == [0, 1, 2, 3]


def test_taken_index_falls_back_to_existing_rule() -> None:
    study = _one_to_one_study()
    runs = _by_path(study=study)
    jobs = [
        _job_record(job_id='41', runs=[runs['pretrain/a']], first_index=5),
        _job_record(job_id='42', runs=[runs['probe/b']], first_index=5),
    ]
    indices = assign_indices(study=study, runs=[runs['pretrain/a'], runs['probe/a']], jobs=jobs)
    assert indices[PurePosixPath('probe/a')] == 6


def test_resubmit_keeps_recorded_indices() -> None:
    study = _one_to_one_study()
    runs = _by_path(study=study)
    jobs = [
        _job_record(job_id='41', runs=[runs['pretrain/a']], first_index=5),
        _job_record(job_id='42', runs=[runs['probe/a']], first_index=2),
    ]
    indices = assign_indices(study=study, runs=[runs['probe/a']], jobs=jobs)
    assert indices[PurePosixPath('probe/a')] == 2


def test_corr_candidate_flag_only_for_one_to_one() -> None:
    for study, expected in ((_one_to_one_study(), True), (_study(), False)):
        runs = _by_path(study=study)
        downstream = next(run for run in runs.values() if run.grid == 'probe')
        upstream = runs[f'pretrain/{downstream.values["dataset"]}']
        targets, _ = plan_targets(
            study=study,
            runs=[downstream],
            reports={upstream.run_path: _report(run=upstream, state=StudyState.RUNNING)},
            submitting=frozenset(),
            jobs=[_job_record(job_id='41', runs=[upstream])],
        )
        assert targets[downstream.run_path][0].corr_ok is expected


def test_aligned_groups_merge_into_one_aftercorr_array(tmp_path: Path) -> None:
    study = _one_to_one_study()
    runs = _by_path(study=study)
    upstream_runs = [runs['pretrain/a'], runs['pretrain/b']]
    downstream_runs = [runs['probe/a'], runs['probe/b']]
    jobs = [_job_record(job_id='52', runs=upstream_runs)]
    reports = {run.run_path: _report(run=run, state=StudyState.RUNNING) for run in upstream_runs}
    targets, _ = plan_targets(
        study=study, runs=downstream_runs, reports=reports, submitting=frozenset(), jobs=jobs
    )
    groups = _plan_groups(study=study, runs=downstream_runs, after=targets, tmp_path=tmp_path)
    assert len(groups) == 1
    (target,) = groups[0].after
    assert (target.corr, target.job_tasks) == (True, ('52',))
    assert dependency_text(after=groups[0].after, upstream={}) == 'aftercorr:52'


def test_misaligned_group_keeps_split_form(tmp_path: Path) -> None:
    study = _one_to_one_study()
    runs = _by_path(study=study)
    upstream_runs = [runs['pretrain/a'], runs['pretrain/b']]
    downstream_runs = [runs['probe/a'], runs['probe/b']]
    jobs = [_job_record(job_id='52', runs=upstream_runs)]
    reports = {run.run_path: _report(run=run, state=StudyState.RUNNING) for run in upstream_runs}
    targets, _ = plan_targets(
        study=study, runs=downstream_runs, reports=reports, submitting=frozenset(), jobs=jobs
    )
    # indices that do not match the upstream ones (0 and 1): probe/a 1, probe/b 0
    machine = MachineProfile(
        path=tmp_path / 'machine.toml', launcher=LauncherKind.LOCAL, output_root=tmp_path / 'runs'
    )
    groups = plan_jobs(
        study=study,
        machine=machine,
        resolved=[
            ResolvedRun(run=run, settings=run.settings, resources=FakeResources())
            for run in downstream_runs
        ],
        indices={PurePosixPath('probe/a'): 1, PurePosixPath('probe/b'): 0},
        extra_args={},
        attempts={},
        first_job_file=1,
        environment={},
        after=targets,
    )
    assert len(groups) == 2
    assert not any(target.corr for group in groups for target in group.after)


def test_same_launch_aligned_groups_resolve_to_the_array_id(tmp_path: Path) -> None:
    study = _one_to_one_study()
    runs = _by_path(study=study)
    selected = [runs['pretrain/a'], runs['pretrain/b'], runs['probe/a'], runs['probe/b']]
    reports = {
        run.run_path: _report(run=run, state=StudyState.MISSING)
        for run in selected
        if run.grid == 'pretrain'
    }
    targets, _ = plan_targets(
        study=study,
        runs=selected,
        reports=reports,
        submitting=frozenset(run.run_path for run in selected),
        jobs=(),
    )
    aligned = {run.run_path: int(run.values['dataset'] == 'b') for run in selected}
    groups = _plan_groups(
        study=study, runs=selected, after=targets, tmp_path=tmp_path, indices=aligned
    )
    probe_groups = [group for group in groups if group.grid == 'probe']
    assert len(probe_groups) == 1
    upstream = {run.run_path: f'77_{position}' for position, run in enumerate(selected[:2])}
    assert dependency_text(after=probe_groups[0].after, upstream=upstream) == 'aftercorr:77'

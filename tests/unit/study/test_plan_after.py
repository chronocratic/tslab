"""Dependency targets, run ordering and target-aware job grouping."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from tests.support.fake_backend import FakeResources
from tslab.core.status import utc_now_iso
from tslab.enums import AfterCondition, JobPhase, LauncherKind, Satisfy, StudyState
from tslab.study import After, Grid, Study
from tslab.study.definition import expand_runs, StudyRun
from tslab.study.launchers.slurm.records import SlurmJobRecord
from tslab.study.machine import MachineProfile
from tslab.study.plan import JobGroup, plan_jobs, ResolvedRun
from tslab.study.plan_after import AfterTarget, order_runs, plan_targets
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


def _study(*, fan_out: bool = False, condition: AfterCondition = AfterCondition.OK) -> Study:
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
        after=(After(grid='pretrain', match=() if fan_out else None, condition=condition),),
    )
    return Study(name='chain', grids={'probe': downstream, 'pretrain': upstream})


def _report(*, run: StudyRun, state: StudyState, job_id: str | None = None) -> RunReport:
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
        indices={run.run_path: position for position, run in enumerate(runs)},
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

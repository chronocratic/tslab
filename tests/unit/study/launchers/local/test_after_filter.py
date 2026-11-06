"""The local launch filter that drops tasks whose upstream dependency is not met."""

import dataclasses
from dataclasses import dataclass
import os
from pathlib import Path

import pytest

from tslab.core.status import InstanceRecord, RunStatusSnapshot, utc_now_iso, write_status
from tslab.enums import InstanceState, RunState
from tslab.study import After, Grid, Study
from tslab.study.context import Context, context_for, Selection
from tslab.study.definition import expand_runs
from tslab.study.launchers.local.after_filter import after_filter
from tslab.study.plan import assign_indices, JobGroup, plan_jobs
from tslab.study.plan_after import plan_targets


def _chain_study() -> Study:
    """Pretrain runs a and b; probe a and b depend on the pretrain of the same dataset."""

    @dataclass(frozen=True, kw_only=True)
    class ChainConfig:
        dataset: str
        data: Path = Path('data')
        output_root: Path = Path('runs')
        run_path: Path = Path('run')

    pretrain = Grid(
        short_name='pre',
        command=('python', 'pre.py'),
        config=ChainConfig,
        axes={'dataset': {'a': {'dataset': 'a'}, 'b': {'dataset': 'b'}}},
    )
    probe = Grid(
        short_name='probe',
        command=('python', 'probe.py'),
        config=ChainConfig,
        axes={'dataset': {'a': {'dataset': 'a'}, 'b': {'dataset': 'b'}}},
        after=(After(grid='pretrain'),),
    )
    return Study(name='chain', grids={'probe': probe, 'pretrain': pretrain})


def _context(*, project: Path, study: Study) -> Context:
    return context_for(
        study=study,
        machine_path=project / 'local.toml',
        selection=Selection(),
        project_root=project,
        local_cpus=1,
        environment=os.environ,
    )


def _groups(*, ctx: Context) -> list[JobGroup]:
    runs = expand_runs(study=ctx.study)
    reports = {**ctx.upstream_reports, **{report.run.run_path: report for report in ctx.reports}}
    targets, _ = plan_targets(
        study=ctx.study,
        runs=runs,
        reports=reports,
        submitting=frozenset(run.run_path for run in runs),
        jobs=ctx.jobs,
        followup=ctx.machine.followup,
    )
    return plan_jobs(
        study=ctx.study,
        machine=ctx.machine,
        resolved=[ctx.resolved[run.run_path] for run in runs],
        indices=assign_indices(study=ctx.study, runs=runs, jobs=ctx.jobs),
        extra_args={},
        attempts={},
        first_job_file=3,
        environment={},
        after=targets,
    )


def _group_of(*, groups: list[JobGroup], run_path: str) -> JobGroup:
    for group in groups:
        if any(task.resolved.run.run_path.as_posix() == run_path for task in group.tasks):
            return group
    message = f'no group with a task of {run_path}'
    raise AssertionError(message)


def _write_pretrain_state(
    *, project: Path, dataset: str, run_state: RunState, instance: InstanceState
) -> None:
    folder = project / 'runs' / 'chain' / 'pretrain' / dataset
    folder.mkdir(parents=True, exist_ok=True)
    write_status(
        path=folder / 'status.json',
        snapshot=RunStatusSnapshot(
            run_state=run_state,
            updated_at=utc_now_iso(),
            instances={'train': InstanceRecord(state=instance)},
        ),
    )


def test_after_filter_passes_groups_without_after(project: Path) -> None:
    ctx = _context(project=project, study=_chain_study())
    group = dataclasses.replace(_group_of(groups=_groups(ctx=ctx), run_path='probe/a'), after=())

    assert after_filter(context=ctx)(group) is group


def test_after_filter_keeps_tasks_when_the_upstream_is_done(project: Path) -> None:
    ctx = _context(project=project, study=_chain_study())
    _write_pretrain_state(
        project=project, dataset='a', run_state=RunState.FINISHED, instance=InstanceState.DONE
    )
    _write_pretrain_state(
        project=project, dataset='b', run_state=RunState.FINISHED, instance=InstanceState.DONE
    )
    group = _group_of(groups=_groups(ctx=ctx), run_path='probe/a')

    assert after_filter(context=ctx)(group) is group


def test_after_filter_drops_tasks_when_the_upstream_failed(
    project: Path, caplog: pytest.LogCaptureFixture
) -> None:
    ctx = _context(project=project, study=_chain_study())
    _write_pretrain_state(
        project=project, dataset='a', run_state=RunState.FAILED, instance=InstanceState.FAILED
    )
    _write_pretrain_state(
        project=project, dataset='b', run_state=RunState.FINISHED, instance=InstanceState.DONE
    )
    groups = _groups(ctx=ctx)
    # why: the targets differ per run, so each probe run is planned as its own group
    group_a = _group_of(groups=groups, run_path='probe/a')
    group_b = _group_of(groups=groups, run_path='probe/b')
    assert group_a is not group_b

    with caplog.at_level('INFO'):
        filtered_a = after_filter(context=ctx)(group_a)
        filtered_b = after_filter(context=ctx)(group_b)

    assert filtered_a.tasks == ()
    assert filtered_b is group_b
    assert 'probe/a' in caplog.text

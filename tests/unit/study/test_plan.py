"""Planning rules: machine checks, local overrides, error text."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.support.fake_backend import FakeResources
from tests.unit.study.configs import grid
from tests.unit.study.helpers import chained_study
from tslab.enums import AfterCondition, ChainedRetry
from tslab.study import After, Grid, setting_axis, Study
from tslab.study.definition import expand_runs, Settings
from tslab.study.machine import LocalSettings, MachineOverride, MachineProfile, SlurmSettings
from tslab.study.machine.profile import FollowupSettings
from tslab.study.plan import (
    _override_warnings,
    _worker_count,
    check_machine,
    expansion_env,
    JobGroup,
    plan_jobs,
    resolve_runs,
    validate_arguments,
    validate_jobs,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


@pytest.fixture
def study() -> Study:
    return Study(name='s', grids={'g': grid()})


def _local(tmp_path: Path, **changes: object) -> MachineProfile:
    values: dict[str, object] = {
        'path': tmp_path / 'm.toml',
        'launcher': 'local',
        'output_root': tmp_path / 'out',
    }
    return MachineProfile(**{**values, **changes})  # ty: ignore[invalid-argument-type]


def test_expansion_env_uses_the_given_environment(tmp_path: Path) -> None:
    machine = _local(tmp_path, env={'ROOT': 'profile-root', 'DATA': 'profile-data'})

    assert expansion_env(machine=machine, environment={'ROOT': 'process-root'}) == {
        'ROOT': 'profile-root',
        'DATA': 'profile-data',
    }


def test_local_machine_with_a_slurm_override_resolves_without_slurm(
    study: Study, tmp_path: Path
) -> None:
    machine = _local(
        tmp_path, overrides=(MachineOverride(match={'recipe': ('cost-paper',)}, slurm={'gpus': 1}),)
    )
    resolved = resolve_runs(
        study=study, machine=machine, runs=expand_runs(study=study), local_cpus=1
    )
    assert all(
        run.resources.describe() == 'local' and run.resources.max_tasks_per_job is None
        for run in resolved
    )


def test_machine_setting_cannot_switch_a_choice(study: Study, tmp_path: Path) -> None:
    machine = _local(tmp_path, settings={'model': 'cost'})
    runs = expand_runs(study=study)
    with pytest.raises(ValueError, match=r'\[settings\] sets model'):
        resolve_runs(study=study, machine=machine, runs=runs, local_cpus=1)


def test_check_machine_raises_for_a_bad_path_and_warns_for_a_dead_override(
    study: Study, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    machine = _local(tmp_path, settings={'trainer.max_epoch': 5})
    with pytest.raises(ValueError, match=r'trainer\.max_epoch'):
        check_machine(study=study, machine=machine)
    machine = _local(tmp_path, overrides=(MachineOverride(match={'nope': ('x',)}, settings={}),))
    warned: set[str] = set()
    with caplog.at_level(logging.WARNING, logger='tslab.study'):
        check_machine(study=study, machine=machine, warned=warned)
        resolve_runs(
            study=study, machine=machine, runs=expand_runs(study=study), warned=warned, local_cpus=1
        )
    assert len([r for r in caplog.records if 'nope' in r.getMessage()]) == 1


def test_override_warnings_returns_texts(study: Study, tmp_path: Path) -> None:
    machine = _local(tmp_path, overrides=(MachineOverride(match={'nope': ('x',)}, settings={}),))
    first = _override_warnings(study=study, machine=machine)
    second = _override_warnings(study=study, machine=machine)
    assert first == second
    assert len(first) == 1
    expected = f'machine override 0 of {machine.path} matches nothing in this study:'
    assert first[0].startswith(expected)
    assert 'nope' in first[0]


@dataclass(frozen=True, kw_only=True)
class _Strict:
    size: int = 1

    def __post_init__(self) -> None:
        if self.size < 0:
            message = 'size must not be negative'
            raise ValueError(message)


def test_validate_arguments_reports_a_post_init_error() -> None:
    error = validate_arguments(config_class=_Strict, arguments=['--size', '-1'])
    assert error is not None
    assert 'size must not be negative' in error


def test_array_size_error_keeps_the_argument_error(study: Study, tmp_path: Path) -> None:
    machine = _local(
        tmp_path,
        launcher='slurm',
        settings={'trainer.accelerator': '{nope}'},
        slurm=SlurmSettings(partition='p', time='1:00:00', max_array_size=1),
    )
    runs = expand_runs(study=study)
    resolved = resolve_runs(study=study, machine=machine, runs=runs, local_cpus=1)
    groups = plan_jobs(
        study=study,
        machine=machine,
        resolved=resolved,
        indices={run.run_path: i for i, run in enumerate(runs)},
        extra_args={},
        attempts={},
        first_job_file=1,
        environment={},
    )
    messages = dict(validate_jobs(study=study, groups=groups))
    assert 'nope' in messages[runs[1].run_path]
    assert 'max_array_size' in messages[runs[1].run_path]


def test_validate_jobs_in_worker_processes_matches_one_process(
    study: Study, tmp_path: Path
) -> None:
    machine = _local(tmp_path)
    runs = expand_runs(study=study)
    resolved = resolve_runs(study=study, machine=machine, runs=runs, local_cpus=1)
    groups = plan_jobs(
        study=study,
        machine=machine,
        resolved=resolved,
        indices={run.run_path: i for i, run in enumerate(runs)},
        extra_args={runs[0].run_path: ('--nope', '1'), runs[2].run_path: ('--seed', 'x')},
        attempts={},
        first_job_file=1,
        environment={},
    )
    serial = validate_jobs(study=study, groups=groups)
    parallel = validate_jobs(study=study, groups=groups, parallel_from=1)
    assert [path for path, _ in serial] == [runs[0].run_path, runs[2].run_path]
    assert parallel == serial


@pytest.mark.parametrize(('cpus', 'workers'), [(1, 1), (2, 1), (12, 6), (13, 6)])
def test_argument_checks_use_at_most_half_the_cpus(
    monkeypatch: pytest.MonkeyPatch, cpus: int, workers: int
) -> None:
    monkeypatch.setattr(os, 'sched_getaffinity', lambda _pid: set(range(cpus)), raising=False)
    assert _worker_count() == workers


def test_fake_resources_have_no_signal_problem() -> None:
    assert FakeResources().signal_problem() is None


def _accelerators_by_recipe(groups: list[JobGroup]) -> dict[str, str]:
    return {
        task.resolved.run.values['recipe']: task.command[
            task.command.index('--trainer.accelerator') + 1
        ]
        for group in groups
        for task in group.tasks
    }


def _plan(study: Study, machine: MachineProfile, *, local_cpus: int = 1) -> list[JobGroup]:
    runs = expand_runs(study=study)
    resolved = resolve_runs(study=study, machine=machine, runs=runs, local_cpus=local_cpus)
    return plan_jobs(
        study=study,
        machine=machine,
        resolved=resolved,
        indices={run.run_path: i for i, run in enumerate(runs)},
        extra_args={},
        attempts={},
        first_job_file=1,
        environment={},
    )


def test_cpus_placeholder_is_the_cpus_per_task_of_the_run(study: Study, tmp_path: Path) -> None:
    machine = _local(
        tmp_path,
        launcher='slurm',
        settings={'trainer.accelerator': '{cpus}'},
        slurm=SlurmSettings(partition='p', cpus_per_task=6),
        overrides=(MachineOverride(match={'recipe': ('cost-paper',)}, slurm={'cpus_per_task': 8}),),
    )
    assert _accelerators_by_recipe(_plan(study, machine)) == {
        'ts2vec-paper': '6',
        'cost-paper': '8',
    }


def test_override_sbatch_extra_merges_key_by_key(study: Study, tmp_path: Path) -> None:
    machine = _local(
        tmp_path,
        launcher='slurm',
        slurm=SlurmSettings(partition='p', sbatch_extra=(('account', 'X'), ('constraint', 'v100'))),
        overrides=(
            MachineOverride(
                match={'recipe': ('cost-paper',)}, slurm={'sbatch_extra': (('constraint', 'a100'),)}
            ),
        ),
    )
    runs = expand_runs(study=study)
    resolved = resolve_runs(study=study, machine=machine, runs=runs, local_cpus=1)
    cost = next(run for run in resolved if run.run.values['recipe'] == 'cost-paper')
    assert isinstance(cost.resources, SlurmSettings)
    assert cost.resources.sbatch_extra == (('account', 'X'), ('constraint', 'a100'))


@pytest.mark.parametrize(
    ('slurm_override', 'expected_problem'),
    [
        (
            {'time': '0:01:00'},
            (
                'slurm.signal_seconds (120) must be smaller than slurm.time (0:01:00); '
                'lower signal_seconds or set it to 0'
            ),
        ),
        ({'time': '0:01:00', 'signal_seconds': 0}, None),
    ],
)
def test_validate_jobs_checks_slurm_signal_seconds_against_override_time(
    study: Study, tmp_path: Path, slurm_override: dict[str, object], expected_problem: str | None
) -> None:
    machine = _local(
        tmp_path,
        launcher='slurm',
        slurm=SlurmSettings(partition='p'),
        overrides=(MachineOverride(match={'grid': ('g',)}, slurm=slurm_override),),
    )
    groups = _plan(study, machine)
    problems = validate_jobs(study=study, groups=groups)
    signal_problems = [
        (path, message) for path, message in problems if 'slurm.signal_seconds' in message
    ]
    if expected_problem is None:
        assert signal_problems == []
        return
    assert signal_problems == [
        (
            groups[0].tasks[0].resolved.run.run_path,
            f'job file {groups[0].job_file}: {expected_problem}',
        )
    ]


def test_cpus_placeholder_on_a_local_machine_is_the_usable_cpus(
    study: Study, tmp_path: Path
) -> None:
    machine = _local(tmp_path, settings={'trainer.accelerator': '{cpus}'})
    assert set(_accelerators_by_recipe(_plan(study, machine, local_cpus=3)).values()) == {'3'}


def test_cpus_per_run_placeholder_expands(study: Study, tmp_path: Path) -> None:
    machine = _local(
        tmp_path,
        local=LocalSettings(max_parallel=2),
        settings={'trainer.accelerator': '{cpus_per_run}'},
    )
    assert set(_accelerators_by_recipe(_plan(study, machine, local_cpus=4)).values()) == {'2'}


@dataclass(frozen=True, kw_only=True)
class _ChainConfig:
    seed: int = 0
    checkpoint: Path = Path()
    output_root: Path = Path('runs')
    run_path: Path = Path('run')


def _chained_study(*, fan_out: bool = False, checkpoint: str = '{after:pretrain}/train') -> Study:
    pretrain = Grid(
        short_name='pre',
        command=('python', '-m', 'pretrain'),
        config=_ChainConfig,
        axes={'dataset': {'a': {}, 'b': {}}, 'seed': setting_axis(path='seed', values=(0, 1))},
        run_path='{grid}/{dataset}/seed_{seed}',
    )
    probe_axes: dict[str, Mapping[str, Settings]] = {
        'dataset': {'a': {}, 'b': {}},
        'head': {'svm': {}},
    }
    probe_path = '{grid}/{dataset}/{head}'
    match = ()
    if not fan_out:
        probe_axes = {
            'dataset': {'a': {}, 'b': {}},
            'seed': setting_axis(path='seed', values=(0, 1)),
            'head': {'svm': {}},
        }
        probe_path = '{grid}/{dataset}/seed_{seed}/{head}'
        match = None
    probe = Grid(
        short_name='probe',
        command=('python', '-m', 'probe'),
        config=_ChainConfig,
        axes=probe_axes,
        after=(After(grid='pretrain', match=match),),
        run_settings={'checkpoint': checkpoint},
        run_path=probe_path,
    )
    return Study(name='chain', grids={'pretrain': pretrain, 'probe': probe})


@pytest.mark.parametrize('checkpoint', ['{after:pretrain}/train', '{after}/train'])
def test_plan_jobs_expands_after_placeholder_for_each_matched_run(
    tmp_path: Path, checkpoint: str
) -> None:
    study = _chained_study(checkpoint=checkpoint)
    machine = _local(tmp_path)
    tasks = [
        task
        for group in _plan(study, machine)
        for task in group.tasks
        if task.resolved.run.grid == 'probe'
    ]
    expected = [
        str(tmp_path / 'out' / 'chain' / 'pretrain' / 'a' / 'seed_0' / 'train'),
        str(tmp_path / 'out' / 'chain' / 'pretrain' / 'a' / 'seed_1' / 'train'),
        str(tmp_path / 'out' / 'chain' / 'pretrain' / 'b' / 'seed_0' / 'train'),
        str(tmp_path / 'out' / 'chain' / 'pretrain' / 'b' / 'seed_1' / 'train'),
    ]

    assert len(tasks) == 4
    assert [task.command[task.command.index('--checkpoint') + 1] for task in tasks] == expected


def test_plan_jobs_reports_fan_out_after_placeholder_error(tmp_path: Path) -> None:
    study = _chained_study(fan_out=True)
    machine = _local(tmp_path)
    tasks = [
        task
        for group in _plan(study, machine)
        for task in group.tasks
        if task.resolved.run.grid == 'probe'
    ]

    assert tasks[0].error == (
        "run probe/a/svm matches 4 runs of grid 'pretrain'; {after:pretrain} needs exactly one"
    )


def test_plan_jobs_reports_bare_after_placeholder_error_for_fan_out(tmp_path: Path) -> None:
    study = _chained_study(fan_out=True, checkpoint='{after}/train')
    machine = _local(tmp_path)
    tasks = [
        task
        for group in _plan(study, machine)
        for task in group.tasks
        if task.resolved.run.grid == 'probe'
    ]

    assert tasks[0].error == (
        "run probe/a/svm matches 4 runs of grid 'pretrain'; {after} needs exactly one"
    )


@pytest.mark.parametrize('chained_retry', [ChainedRetry.RESUBMIT, ChainedRetry.REQUEUE])
def test_check_machine_rejects_failed_condition_with_chained_retry(
    tmp_path: Path, chained_retry: ChainedRetry
) -> None:
    study = chained_study(condition=AfterCondition.FAILED)
    machine = _local(tmp_path, followup=FollowupSettings(chained_retry=chained_retry))
    with pytest.raises(ValueError, match=r"grid 'probe'.*condition 'failed'.*chained_retry"):
        check_machine(study=study, machine=machine)


def test_check_machine_accepts_failed_condition_with_manual_chained_retry(tmp_path: Path) -> None:
    study = chained_study(condition=AfterCondition.FAILED)
    check_machine(study=study, machine=_local(tmp_path))
    ok = chained_study(condition=AfterCondition.OK)
    check_machine(
        study=ok,
        machine=_local(tmp_path, followup=FollowupSettings(chained_retry=ChainedRetry.RESUBMIT)),
    )

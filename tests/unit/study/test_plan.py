"""Planning rules found in review: machine checks, local overrides, error text."""

from dataclasses import dataclass
import logging
from pathlib import Path

import pytest

from study.configs import grid
from tslab.study import Study
from tslab.study.definition import expand_runs
from tslab.study.machine import MachineOverride, MachineProfile, SlurmSettings
from tslab.study.plan import (
    check_machine,
    plan_jobs,
    resolve_runs,
    validate_arguments,
    validate_jobs,
)


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


def test_local_machine_with_a_slurm_override_resolves_without_slurm(
    study: Study, tmp_path: Path
) -> None:
    machine = _local(
        tmp_path, overrides=(MachineOverride(match={'recipe': ('cost-paper',)}, slurm={'gpus': 1}),)
    )
    resolved = resolve_runs(study=study, machine=machine, runs=expand_runs(study=study))
    assert all(run.slurm is None for run in resolved)


def test_machine_setting_that_switches_a_choice_drops_stale_children(
    study: Study, tmp_path: Path
) -> None:
    machine = _local(tmp_path, settings={'model': 'cost'})
    runs = expand_runs(study=study)
    assert any('model.depth' in run.settings for run in runs)
    resolved = resolve_runs(study=study, machine=machine, runs=runs)
    ts2vec = next(run for run in resolved if run.run.values['recipe'] == 'ts2vec-paper')
    assert ts2vec.settings['model'] == 'cost'
    assert not any(path.startswith('model.') for path in ts2vec.settings)


def test_check_machine_raises_for_a_bad_path_and_warns_for_a_dead_override(
    study: Study, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    machine = _local(tmp_path, settings={'trainer.max_epoch': 5})
    with pytest.raises(ValueError, match=r'trainer\.max_epoch'):
        check_machine(study=study, machine=machine)
    machine = _local(tmp_path, overrides=(MachineOverride(match={'nope': ('x',)}, settings={}),))
    with caplog.at_level(logging.WARNING, logger='tslab.study'):
        check_machine(study=study, machine=machine)
        resolve_runs(study=study, machine=machine, runs=expand_runs(study=study))
    assert len([r for r in caplog.records if 'nope' in r.getMessage()]) == 1


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
        settings={'trainer.max_training_time': '{nope}'},
        slurm=SlurmSettings(partition='p', time='1:00:00', max_array_size=1),
    )
    runs = expand_runs(study=study)
    resolved = resolve_runs(study=study, machine=machine, runs=runs)
    groups = plan_jobs(
        study=study,
        machine=machine,
        resolved=resolved,
        indices={run.run_path: i for i, run in enumerate(runs)},
        extra_args={},
        attempts={},
        first_job_file=1,
    )
    messages = dict(validate_jobs(study=study, groups=groups))
    assert 'nope' in messages[runs[1].run_path]
    assert 'max_array_size' in messages[runs[1].run_path]

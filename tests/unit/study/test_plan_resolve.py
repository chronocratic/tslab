"""``resolve_runs``: the settings and resources of each run on one machine."""

import logging
import os
from pathlib import Path, PurePosixPath

import pytest

from tests.unit.study.configs import grid
from tslab.study import Study
from tslab.study.definition import expand_runs
from tslab.study.machine import load_machine, MachineProfile, SlurmSettings
from tslab.study.plan import resolve_runs

BASE_TOML = (
    'launcher = "slurm"\noutput_root = "runs"\n[slurm]\npartition = "p"\ntime = "1-00:00:00"\n'
)


@pytest.fixture
def study() -> Study:
    return Study(name='s', grids={'g': grid()})


def _profile(tmp_path: Path, text: str) -> MachineProfile:
    path = tmp_path / 'm.toml'
    path.write_text(BASE_TOML + text)
    return load_machine(path=path, environment=os.environ)


def test_merge_order_and_slurm_precedence(study: Study, tmp_path: Path) -> None:
    profile = _profile(
        tmp_path,
        '[settings]\n"trainer.accelerator" = "gpu"\n'
        '[[overrides]]\nmatch = { recipe = "cost-paper" }\n'
        'slurm = { gpus = 1, time = "14-00:00:00" }\n'
        '[[overrides]]\nmatch = { recipe = "cost-paper", dataset = "Coffee" }\n'
        'slurm = { time = "1-00:00:00" }\nsettings = { "trainer.accelerator" = "cpu" }\n',
    )
    resolved = {
        r.run.run_path: r
        for r in resolve_runs(
            study=study, machine=profile, runs=expand_runs(study=study), local_cpus=1
        )
    }
    cost = resolved[PurePosixPath('g/cost-paper/Coffee/42')]
    assert isinstance(cost.resources, SlurmSettings)
    assert cost.resources.gpus == 1
    assert cost.resources.time == '1-00:00:00'
    assert cost.settings['trainer.accelerator'] == 'cpu'
    ts2vec = resolved[PurePosixPath('g/ts2vec-paper/Coffee/42')]
    assert ts2vec.resources == profile.slurm
    assert ts2vec.settings['trainer.accelerator'] == 'gpu'


def test_profile_experiment_setting_rejected(study: Study, tmp_path: Path) -> None:
    profile = _profile(tmp_path, '[settings]\n"trainer.max_epochs" = 5\n')
    with pytest.raises(
        ValueError,
        match=(
            r'\[settings\] sets trainer\.max_epochs, which is not a machine setting of '
            r'.* \(grid .*\); set it in the study instead'
        ),
    ):
        resolve_runs(study=study, machine=profile, runs=expand_runs(study=study), local_cpus=1)


def test_override_experiment_setting_rejected(study: Study, tmp_path: Path) -> None:
    profile = _profile(
        tmp_path, '[[overrides]]\nmatch = { grid = "g" }\nsettings = { "trainer.max_epochs" = 5 }\n'
    )
    with pytest.raises(ValueError, match=r'overrides\[0\]\.settings sets trainer\.max_epochs'):
        resolve_runs(study=study, machine=profile, runs=expand_runs(study=study), local_cpus=1)


def test_every_bad_key_is_listed(study: Study, tmp_path: Path) -> None:
    profile = _profile(
        tmp_path, '[settings]\n"trainer.max_epochs" = 5\n"trainer.max_training_time" = "6:00:00"\n'
    )
    with pytest.raises(
        ValueError, match=r'trainer\.max_epochs, trainer\.max_training_time'
    ) as raised:
        resolve_runs(study=study, machine=profile, runs=expand_runs(study=study), local_cpus=1)
    assert 'trainer.max_epochs' in str(raised.value)
    assert 'trainer.max_training_time' in str(raised.value)


def test_unknown_names_in_machine_overrides_warn(
    study: Study, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    profile = _profile(
        tmp_path,
        '[[overrides]]\nmatch = { recipe = ["cost-paper", "old-recipe"] }\nslurm = { gpus = 1 }\n'
        '[[overrides]]\nmatch = { variant = "ETTh1" }\nslurm = { gpus = 2 }\n',
    )
    with caplog.at_level(logging.WARNING, logger='tslab.study'):
        resolved = resolve_runs(
            study=study, machine=profile, runs=expand_runs(study=study), local_cpus=1
        )
    assert 'old-recipe' in caplog.text
    assert 'variant' in caplog.text
    assert any(
        isinstance(run.resources, SlurmSettings) and run.resources.gpus == 1 for run in resolved
    )

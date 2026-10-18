"""``resolve_runs``: the settings and resources of each run on one machine."""

import logging
from pathlib import Path

import pytest

from study.configs import grid
from tslab.study import expand_runs, load_machine, MachineProfile, resolve_runs, Study

BASE_TOML = (
    'launcher = "slurm"\noutput_root = "runs"\n[slurm]\npartition = "p"\ntime = "1-00:00:00"\n'
)


@pytest.fixture
def study() -> Study:
    return Study(name='s', grids={'g': grid()})


def _profile(tmp_path: Path, text: str) -> MachineProfile:
    path = tmp_path / 'm.toml'
    path.write_text(BASE_TOML + text)
    return load_machine(path=path)


def test_merge_order_and_slurm_precedence(study: Study, tmp_path: Path) -> None:
    profile = _profile(
        tmp_path,
        '[settings]\n"trainer.max_epochs" = 5\n'
        '[[overrides]]\nmatch = { recipe = "cost-paper" }\n'
        'slurm = { gpus = 1, time = "14-00:00:00" }\n'
        '[[overrides]]\nmatch = { recipe = "cost-paper", dataset = "Coffee" }\n'
        'slurm = { time = "1-00:00:00" }\nsettings = { "trainer.max_epochs" = 1 }\n',
    )
    resolved = {
        r.run.run_path: r
        for r in resolve_runs(study=study, machine=profile, runs=expand_runs(study=study))
    }
    cost = resolved['g/cost-paper/Coffee/42']
    assert cost.slurm is not None
    assert cost.slurm.gpus == 1
    assert cost.slurm.time == '1-00:00:00'
    assert cost.settings['trainer.max_epochs'] == 1
    ts2vec = resolved['g/ts2vec-paper/Coffee/42']
    assert ts2vec.slurm == profile.slurm
    assert ts2vec.settings['trainer.max_epochs'] == 5


def test_unknown_names_in_machine_overrides_warn(
    study: Study, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    profile = _profile(
        tmp_path,
        '[[overrides]]\nmatch = { recipe = ["cost-paper", "old-recipe"] }\nslurm = { gpus = 1 }\n'
        '[[overrides]]\nmatch = { variant = "ETTh1" }\nslurm = { gpus = 2 }\n',
    )
    with caplog.at_level(logging.WARNING, logger='tslab.study'):
        resolved = resolve_runs(study=study, machine=profile, runs=expand_runs(study=study))
    assert 'old-recipe' in caplog.text
    assert 'variant' in caplog.text
    assert any(r.slurm is not None and r.slurm.gpus == 1 for r in resolved)


def test_unknown_machine_setting_path_raises(study: Study, tmp_path: Path) -> None:
    profile = _profile(tmp_path, '[settings]\n"trainer.max_epoch" = 5\n')
    with pytest.raises(ValueError, match=r'trainer\.max_epoch'):
        resolve_runs(study=study, machine=profile, runs=expand_runs(study=study))

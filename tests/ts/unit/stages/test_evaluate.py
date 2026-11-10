from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from tests.ts.unit.stage_fakes import run_in
from tests.ts.unit.stages.helpers import encoded
import tslab.lab
import tslab.ts
from tslab.ts.probes import SVMProbe


def test_evaluate_fits_the_probe_and_logs_its_metrics(tmp_path: Path) -> None:
    with run_in(root=tmp_path) as run:
        metrics = tslab.ts.stages.evaluate(
            run=run, representations=encoded(run=run), downstream_model=SVMProbe(n_jobs=1)
        )
    assert metrics.values['accuracy'] == 1.0
    assert set(metrics.values) == {'accuracy', 'precision', 'recall', 'f1'}
    logged = (tmp_path / 'r' / 'trackers' / 'csv' / 'version_0' / 'metrics.csv').read_text()
    assert 'evaluate/default/accuracy' in logged


@dataclass(frozen=True, kw_only=True)
class Agreement:
    name: str = 'agreement'

    def __call__(self, *, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        return float((predictions == targets).float().mean())


def test_evaluate_uses_the_given_metrics(tmp_path: Path) -> None:
    with run_in(root=tmp_path) as run:
        metrics = tslab.ts.stages.evaluate(
            run=run,
            name='custom',
            representations=encoded(run=run),
            downstream_model=SVMProbe(n_jobs=1),
            metrics=[Agreement()],
        )
    assert metrics.values == {'agreement': 1.0}


@tslab.lab.stage(name='partial', output=tslab.lab.Tensors)
def partial(*, run: tslab.lab.Run) -> tslab.lab.Tensors:
    del run
    return tslab.lab.Tensors(tensors={'train_representations': torch.zeros(2, 2)})


def test_evaluate_names_missing_keys(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match='train_targets'), run_in(root=tmp_path) as run:
        tslab.ts.stages.evaluate(
            run=run, representations=partial(run=run), downstream_model=SVMProbe(n_jobs=1)
        )

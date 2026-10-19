from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from tests.unit.stage_fakes import FakeTrainer, LinearEncoder, run_in, ToySplits
import tslab
from tslab.probes import SVMProbe
from tslab.stages import check_representations


def test_train_writes_its_checkpoint_into_the_instance_folder(tmp_path: Path) -> None:
    with run_in(root=tmp_path) as run:
        checkpoint = tslab.stages.train(
            run=run, trainer=FakeTrainer(), model=LinearEncoder(), data=ToySplits()
        )
    folder = tmp_path / 'r' / 'train' / 'default'
    assert checkpoint.directory == folder
    assert sorted(path.name for path in folder.iterdir()) == [
        'checkpoint.json',
        'last.ckpt',
        'meta.json',
    ]


def test_finished_training_is_loaded_not_repeated(tmp_path: Path) -> None:
    with run_in(root=tmp_path) as run:
        tslab.stages.train(run=run, trainer=FakeTrainer(), model=LinearEncoder(), data=ToySplits())
    trainer = FakeTrainer()
    with run_in(root=tmp_path) as run:
        checkpoint = tslab.stages.train(
            run=run, trainer=trainer, model=LinearEncoder(), data=ToySplits()
        )
    assert trainer.fit_calls == 0
    assert checkpoint.best == 'last.ckpt'


def test_failed_training_resumes_in_the_same_folder(tmp_path: Path) -> None:
    with run_in(root=tmp_path, fail_fast=False) as run:
        failed = tslab.stages.train(
            run=run, trainer=FakeTrainer(fail=True), model=LinearEncoder(), data=ToySplits()
        )
    assert not tslab.is_available(artifact=failed)
    assert (tmp_path / 'r' / 'train' / 'default' / '.incomplete').is_file()
    trainer = FakeTrainer()
    with run_in(root=tmp_path) as run:
        tslab.stages.train(run=run, trainer=trainer, model=LinearEncoder(), data=ToySplits())
    assert trainer.resumed
    assert not (tmp_path / 'r' / 'train' / 'default' / '.incomplete').exists()


def encoded(*, run: tslab.Run, model: LinearEncoder | None = None) -> tslab.Tensors:
    model = LinearEncoder() if model is None else model
    trainer = FakeTrainer()
    checkpoint = tslab.stages.train(run=run, trainer=trainer, model=model, data=ToySplits())
    return tslab.stages.encode(
        run=run,
        trainer=trainer,
        model=model,
        checkpoint=checkpoint,
        data=ToySplits(),
        encoder=tslab.ModelAsEncoder(device='cpu'),
    )


def test_encode_never_uses_the_training_loaders(tmp_path: Path) -> None:
    with run_in(root=tmp_path) as run:
        representations = encoded(run=run)
    shapes = {key: tuple(value.shape) for key, value in representations.tensors.items()}
    assert shapes == {
        'train_representations': (40, 4),
        'train_targets': (40,),
        'val_representations': (20, 4),
        'val_targets': (20,),
        'test_representations': (20, 4),
        'test_targets': (20,),
    }
    assert torch.equal(
        representations.tensors['test_targets'], ToySplits().labeled_split(split='test')[1]
    )


def test_dead_encoder_fails_the_instance(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='non-finite'), run_in(root=tmp_path) as run:
        encoded(run=run, model=LinearEncoder(dead=True))


def test_check_representations_rejects_wrong_shapes() -> None:
    with pytest.raises(ValueError, match='2-D'):
        check_representations(
            representations=torch.zeros(3, 2, 2), expected_rows=3, description='x'
        )
    with pytest.raises(ValueError, match='rows'):
        check_representations(representations=torch.zeros(2, 4), expected_rows=3, description='x')
    check_representations(representations=torch.zeros(0, 4), expected_rows=0, description='x')
    some_nan = torch.zeros(10, 4)
    some_nan[0] = torch.nan
    check_representations(representations=some_nan, expected_rows=10, description='x')


def test_encode_requires_a_fingerprintable_encoder(tmp_path: Path) -> None:
    model, trainer = LinearEncoder(), FakeTrainer()
    # why: the train call sets up the encode call that must fail
    with pytest.raises(TypeError, match='encoder'), run_in(root=tmp_path) as run:  # noqa: PT012
        checkpoint = tslab.stages.train(run=run, trainer=trainer, model=model, data=ToySplits())
        tslab.stages.encode(
            run=run,
            trainer=trainer,
            model=model,
            checkpoint=checkpoint,
            data=ToySplits(),
            encoder=lambda model: model,
        )


def test_evaluate_fits_the_probe_and_logs_its_metrics(tmp_path: Path) -> None:
    with run_in(root=tmp_path) as run:
        metrics = tslab.stages.evaluate(
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
        metrics = tslab.stages.evaluate(
            run=run,
            name='custom',
            representations=encoded(run=run),
            downstream_model=SVMProbe(n_jobs=1),
            metrics=[Agreement()],
        )
    assert metrics.values == {'agreement': 1.0}


@tslab.stage(name='partial', output=tslab.Tensors)
def partial(*, run: tslab.Run) -> tslab.Tensors:
    del run
    return tslab.Tensors(tensors={'train_representations': torch.zeros(2, 2)})


def test_evaluate_names_missing_keys(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match='train_targets'), run_in(root=tmp_path) as run:
        tslab.stages.evaluate(
            run=run, representations=partial(run=run), downstream_model=SVMProbe(n_jobs=1)
        )

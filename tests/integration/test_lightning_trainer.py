from collections.abc import Generator
import contextlib
import dataclasses
from dataclasses import dataclass
import logging
from pathlib import Path

from lightning_toy import FAST, ToyData, ToyDataModule, ToyInfo, ToyRegressor
from mlflow.tracking.client import MlflowClient
import pytest
import torch

from tslab.core.artifact import Checkpoint
from tslab.core.run import Run
from tslab.interfaces.tracker import Tracker
from tslab.lightning import LightningDataSource, LightningTrainer, TrainingTimeLimitReached
from tslab.tracking.mlflow_tracker import MLflowTracker


@dataclass(frozen=True, kw_only=True)
class Config:
    seed: int = 0


@contextlib.contextmanager
def run_in(*, root: Path, trackers: tuple[Tracker, ...] = ()) -> Generator[Run]:
    with Run(config=Config(), root=root, relative_path=Path('r'), trackers=trackers) as run:
        yield run


def trainer(**changes: object) -> LightningTrainer:
    return LightningTrainer(config=dataclasses.replace(FAST, **changes))


def test_fit_returns_a_checkpoint_naming_best_and_last(tmp_path: Path) -> None:
    with run_in(root=tmp_path) as run:
        directory = run.directory / 'train'
        checkpoint = trainer().fit(
            run=run, model=ToyRegressor(), data=ToyData(), checkpoint_directory=directory
        )
    assert checkpoint.last == 'last.ckpt'
    assert checkpoint.best.startswith('epoch=')
    assert 'val_loss_epoch=' in checkpoint.best
    assert checkpoint.best_path.is_file()
    checkpoint.save(directory=directory)
    assert Checkpoint.load(directory=directory).best == checkpoint.best
    # the metrics of Lightning reached the always-on CSV tracker of the run.
    metrics = (tmp_path / 'r' / 'trackers' / 'csv' / 'version_0' / 'metrics.csv').read_text()
    assert 'train_loss_epoch' in metrics


def test_load_restores_the_best_weights(tmp_path: Path) -> None:
    lightning_trainer = trainer()
    with run_in(root=tmp_path) as run:
        checkpoint = lightning_trainer.fit(
            run=run,
            model=ToyRegressor(),
            data=ToyData(),
            checkpoint_directory=run.directory / 'train',
        )
    fresh = ToyRegressor()
    lightning_trainer.load(model=fresh, checkpoint=checkpoint)
    best = torch.load(f=checkpoint.best_path, map_location='cpu', weights_only=False)
    assert torch.equal(fresh.layer.weight, best['state_dict']['layer.weight'])


def test_without_validation_last_is_best(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING), run_in(root=tmp_path) as run:
        checkpoint = trainer().fit(
            run=run,
            model=ToyRegressor(),
            data=ToyData(with_validation=False),
            checkpoint_directory=run.directory / 'train',
        )
    assert checkpoint.best == checkpoint.last == 'last.ckpt'
    assert 'no validation loader' in caplog.text


def test_time_limit_raises_and_resubmission_resumes(tmp_path: Path) -> None:
    directory = tmp_path / 'r' / 'train'
    # A zero limit stops after the first batch, like a job whose time ran out.
    with pytest.raises(TrainingTimeLimitReached, match='resubmit'), run_in(root=tmp_path) as run:
        trainer(max_training_time='00:00:00').fit(
            run=run, model=ToyRegressor(), data=ToyData(), checkpoint_directory=directory
        )
    with run_in(root=tmp_path) as run:
        checkpoint = trainer().fit(
            run=run, model=ToyRegressor(), data=ToyData(), checkpoint_directory=directory
        )
    assert checkpoint.last_path.is_file()


def test_min_epochs_with_a_time_limit_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING), run_in(root=tmp_path) as run:
        trainer(min_epochs=1, max_training_time='01:00:00').fit(
            run=run,
            model=ToyRegressor(),
            data=ToyData(),
            checkpoint_directory=run.directory / 'train',
        )
    assert '--trainer.min-epochs 0' in caplog.text


def test_training_error_propagates_without_retries(tmp_path: Path) -> None:
    model = ToyRegressor(fail_at_step=3)
    with pytest.raises(RuntimeError, match='out of memory'), run_in(root=tmp_path) as run:
        trainer().fit(
            run=run, model=model, data=ToyData(), checkpoint_directory=run.directory / 'train'
        )
    assert model.fit_starts == 1


class Marker:
    """A custom class: safe loading refuses to unpickle it."""


def test_untrusted_external_checkpoint_refuses_objects(tmp_path: Path) -> None:
    path = tmp_path / 'downloaded.ckpt'
    torch.save(obj={'state_dict': ToyRegressor().state_dict(), 'extra': Marker()}, f=path)
    lightning_trainer = trainer()
    with pytest.raises(ValueError, match=r'trusted=True'):
        lightning_trainer.load(model=ToyRegressor(), checkpoint=Checkpoint.external(path=path))
    loaded = ToyRegressor()
    lightning_trainer.load(model=loaded, checkpoint=Checkpoint.external(path=path, trusted=True))
    saved = torch.load(f=path, weights_only=False)['state_dict']
    assert torch.equal(loaded.layer.weight, saved['layer.weight'])


def test_tracker_status_is_set_by_the_run_not_by_lightning(tmp_path: Path) -> None:
    tracker = MLflowTracker(experiment='m4')

    def script() -> None:
        with run_in(root=tmp_path, trackers=(tracker,)) as run:
            trainer().fit(
                run=run,
                model=ToyRegressor(),
                data=ToyData(),
                checkpoint_directory=run.directory / 'train',
            )
            # Lightning already called finalize('success') on the MLflow logger.
            message = 'failure after training'
            raise RuntimeError(message)

    with pytest.raises(RuntimeError, match='after training'):
        script()
    database = (tmp_path / 'r' / 'trackers' / 'mlflow' / 'mlflow.db').resolve()
    client = MlflowClient(tracking_uri=f'sqlite:///{database.as_posix()}')
    experiment = client.get_experiment_by_name('m4')
    assert experiment is not None
    (mlflow_run,) = client.search_runs(experiment_ids=[experiment.experiment_id])
    assert mlflow_run.info.status == 'FAILED'
    assert 'train_loss_epoch' in mlflow_run.data.metrics


def test_lightning_data_source_sets_up_lazily_and_once(tmp_path: Path) -> None:
    datamodule = ToyDataModule()
    data = LightningDataSource(datamodule=datamodule, info=ToyInfo(num_features=4))
    assert data.info == ToyInfo(num_features=4)
    assert data.has_validation is False
    assert data.val_loader() is None
    assert datamodule.calls == []
    data.train_loader()
    data.train_loader()
    data.test_loader()
    assert datamodule.calls == ['prepare', 'setup:fit', 'setup:test']


def test_fit_with_a_datamodule(tmp_path: Path) -> None:
    data = LightningDataSource(datamodule=ToyDataModule(), info=ToyInfo(num_features=4))
    with run_in(root=tmp_path) as run:
        checkpoint = trainer().fit(
            run=run, model=ToyRegressor(), data=data, checkpoint_directory=run.directory / 'train'
        )
    assert checkpoint.best == checkpoint.last == 'last.ckpt'

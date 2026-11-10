import dataclasses
import logging
from pathlib import Path
import time
from typing import override

from lightning.pytorch.plugins.environments import SLURMEnvironment
from mlflow.tracking.client import MlflowClient
import pytest
import torch

from tests.ml.integration.lightning_toy import FAST, ToyData, ToyDataModule, ToyInfo, ToyRegressor
from tests.support.integration_helpers import run_in
from tslab.ml.artifacts import Checkpoint
from tslab.ml.lightning import LightningDataSource, LightningTrainer, TrainingPaused
from tslab.ml.lightning.config import LightningTrainerConfig
from tslab.ml.tracking.mlflow_tracker import MLflowTracker


def trainer(**changes: object) -> LightningTrainer:
    return LightningTrainer(config=dataclasses.replace(FAST, **changes))


@pytest.mark.slow
def test_trainer_turns_off_auto_requeue_under_slurm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for key, value in {
        'SLURM_JOB_ID': '4242',
        'SLURM_NTASKS': '1',
        'SLURM_JOB_NAME': 'tslab-test',
        'SLURM_NODELIST': 'localhost',
        'SLURM_PROCID': '0',
        'SLURM_LOCALID': '0',
        'SLURM_NODEID': '0',
    }.items():
        monkeypatch.setenv(key, value)
    with run_in(root=tmp_path) as run:
        built = trainer()._build_trainer(  # noqa: SLF001 # why: inspect the selected plugin without fitting
            run=run,
            checkpoint_directory=run.directory / 'train',
            has_validation=False,
            max_time=None,
        )
        environment = built._accelerator_connector.cluster_environment  # noqa: SLF001 # why: Lightning stores the plugin here
    assert isinstance(environment, SLURMEnvironment)
    assert environment.auto_requeue is False


def test_extra_trainer_kwargs_cannot_set_plugins() -> None:
    with pytest.raises(ValueError, match='plugins'):
        LightningTrainer(config=LightningTrainerConfig(extra_trainer_kwargs={'plugins': 'env'}))


@pytest.mark.slow
def test_fit_returns_a_checkpoint_naming_best_and_last(tmp_path: Path) -> None:
    lightning_trainer = trainer()
    with run_in(root=tmp_path) as run:
        directory = run.directory / 'train'
        checkpoint = lightning_trainer.fit(
            run=run, model=ToyRegressor(), data=ToyData(), checkpoint_directory=directory
        )
    assert checkpoint.last == 'last.ckpt'
    assert checkpoint.best.startswith('epoch=')
    assert 'val_loss_epoch=' in checkpoint.best
    assert checkpoint.best_path.is_file()
    checkpoint.save(directory=directory)
    assert Checkpoint.load(directory=directory).best == checkpoint.best
    # The Lightning metrics reached the always-on CSV tracker of the run.
    metrics = (tmp_path / 'r' / 'trackers' / 'csv' / 'version_0' / 'metrics.csv').read_text()
    assert 'train_loss_epoch' in metrics
    fresh = ToyRegressor()
    lightning_trainer.load(model=fresh, checkpoint=checkpoint)
    best = torch.load(f=checkpoint.best_path, map_location='cpu', weights_only=False)
    assert torch.equal(fresh.layer.weight, best['state_dict']['layer.weight'])


@pytest.mark.slow
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


@pytest.mark.slow
def test_max_training_time_finishes_and_last_holds_the_stop_step(tmp_path: Path) -> None:
    with run_in(root=tmp_path) as run:
        checkpoint = trainer(
            max_epochs=500, max_training_time='00:00:01', early_stopping_patience=300
        ).fit(
            run=run,
            model=_SlowToyRegressor(sleep_seconds=0.6),
            data=ToyData(),
            checkpoint_directory=run.directory / 'train',
        )
    state = torch.load(checkpoint.last_path, weights_only=False)
    assert 0 < state['global_step'] < 8


@pytest.mark.slow
def test_max_training_time_counts_earlier_jobs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_changes = {
        'max_epochs': 500,
        'max_training_time': '00:00:01',
        'early_stopping_patience': 300,
    }
    directory = tmp_path / 'r' / 'train'
    monkeypatch.setenv('SLURM_JOB_END_TIME', '0')
    with pytest.raises(TrainingPaused), run_in(root=tmp_path) as run:
        trainer(**config_changes).fit(
            run=run,
            model=_SlowToyRegressor(sleep_seconds=1.1),
            data=ToyData(),
            checkpoint_directory=directory,
        )
    first_step = torch.load(directory / 'last.ckpt', weights_only=False)['global_step']
    monkeypatch.delenv('SLURM_JOB_END_TIME')
    with run_in(root=tmp_path) as run:
        checkpoint = trainer(**config_changes).fit(
            run=run, model=ToyRegressor(), data=ToyData(), checkpoint_directory=directory
        )
    resumed_step = torch.load(checkpoint.last_path, weights_only=False)['global_step']
    assert resumed_step <= first_step + 1


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
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


class _SlowToyRegressor(ToyRegressor):
    def __init__(self, *, sleep_seconds: float) -> None:
        super().__init__()
        self.sleep_seconds = sleep_seconds

    @override
    def training_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = super().training_step(batch, batch_idx)
        time.sleep(self.sleep_seconds)
        return loss


@pytest.mark.slow
def test_fit_with_a_datamodule(tmp_path: Path) -> None:
    data = LightningDataSource(datamodule=ToyDataModule(), info=ToyInfo(num_features=4))
    with run_in(root=tmp_path) as run:
        checkpoint = trainer().fit(
            run=run, model=ToyRegressor(), data=data, checkpoint_directory=run.directory / 'train'
        )
    assert checkpoint.best == checkpoint.last == 'last.ckpt'

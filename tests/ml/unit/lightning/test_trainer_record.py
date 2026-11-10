from collections.abc import Iterable
import logging
from pathlib import Path
import time
from typing import cast, override

from lightning import LightningModule
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from tslab.lab.core.files import read_json_object
from tslab.lab.core.run import Run
from tslab.ml.interfaces import Batch
from tslab.ml.lightning.config import LightningTrainerConfig
from tslab.ml.lightning.exceptions import TrainingPaused
from tslab.ml.lightning.trainer import LightningTrainer


class _ToyData:
    """64 training and 16 validation samples of a noisy linear function, batches of 8."""

    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(0)
        weights = torch.randn(4, 1, generator=generator)
        inputs = torch.randn(80, 4, generator=generator)
        targets = inputs @ weights + 0.1 * torch.randn(80, 1, generator=generator)
        self._train = TensorDataset(inputs[:64], targets[:64])
        self._val = TensorDataset(inputs[64:], targets[64:])

    @property
    def info(self) -> dict[str, int]:
        return {'num_features': 4}

    def train_loader(self) -> Iterable[Batch]:
        return cast(
            'Iterable[Batch]',
            DataLoader(
                dataset=self._train,
                batch_size=8,
                shuffle=True,
                generator=torch.Generator().manual_seed(0),
            ),
        )

    def val_loader(self) -> Iterable[Batch]:
        return cast('Iterable[Batch]', DataLoader(dataset=self._val, batch_size=8))

    def test_loader(self) -> Iterable[Batch]:
        return cast('Iterable[Batch]', DataLoader(dataset=self._val, batch_size=8))


class _ToyModel(LightningModule):
    """A linear layer fit by SGD; a zero learning rate keeps the loss flat."""

    def __init__(self, *, learning_rate: float, seconds_per_batch: float = 0.0) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(in_features=4, out_features=1)
        self.learning_rate = learning_rate
        self.seconds_per_batch = seconds_per_batch

    @override
    def training_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        del batch_idx
        if self.seconds_per_batch:
            time.sleep(self.seconds_per_batch)
        inputs, targets = batch
        loss = torch.nn.functional.mse_loss(input=self.layer(inputs), target=targets)
        self.log(name='train_loss', value=loss, on_step=True, on_epoch=True)
        return loss

    @override
    def validation_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        del batch_idx
        inputs, targets = batch
        loss = torch.nn.functional.mse_loss(input=self.layer(inputs), target=targets)
        self.log(name='val_loss', value=loss, on_step=True, on_epoch=True)
        return loss

    @override
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(params=self.parameters(), lr=self.learning_rate)


def _fit(*, tmp_path: Path, config: LightningTrainerConfig, model: _ToyModel) -> Path:
    """Train ``model`` in ``tmp_path`` and return the checkpoint directory."""
    run = Run(config=LightningTrainerConfig(), root=tmp_path, relative_path=Path('run'))
    trainer = LightningTrainer(config=config)
    checkpoint_directory = tmp_path / 'checkpoints'
    trainer.fit(run=run, model=model, data=_ToyData(), checkpoint_directory=checkpoint_directory)
    return checkpoint_directory


def test_training_record_max_epochs(tmp_path: Path) -> None:
    config = LightningTrainerConfig(
        max_epochs=2,
        min_epochs=0,
        accelerator='cpu',
        devices=1,
        enable_progress_bar=False,
        early_stopping_patience=100,
    )
    checkpoint_directory = _fit(
        tmp_path=tmp_path, config=config, model=_ToyModel(learning_rate=0.05)
    )
    record = read_json_object(path=checkpoint_directory / 'training.json')
    assert record['stop_reason'] == 'max_epochs'
    assert record['epochs'] == 2


def test_training_record_early_stopping(tmp_path: Path) -> None:
    # A zero learning rate keeps the loss flat, so early stopping stops after two checks.
    config = LightningTrainerConfig(
        max_epochs=10,
        min_epochs=0,
        accelerator='cpu',
        devices=1,
        enable_progress_bar=False,
        early_stopping_patience=1,
    )
    checkpoint_directory = _fit(
        tmp_path=tmp_path, config=config, model=_ToyModel(learning_rate=0.0)
    )
    record = read_json_object(path=checkpoint_directory / 'training.json')
    assert record['stop_reason'] == 'early_stopping'


def test_training_record_time_limit_warns(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    config = LightningTrainerConfig(
        max_epochs=10,
        min_epochs=0,
        accelerator='cpu',
        devices=1,
        enable_progress_bar=False,
        max_training_time='0:01',
    )
    with caplog.at_level(logging.WARNING):
        checkpoint_directory = _fit(
            tmp_path=tmp_path,
            config=config,
            model=_ToyModel(learning_rate=0.05, seconds_per_batch=0.1),
        )
    record = read_json_object(path=checkpoint_directory / 'training.json')
    assert record['stop_reason'] == 'max_training_time'
    assert 'training stopped at max_training_time' in caplog.text


def test_paused_training_writes_no_record(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('SLURM_JOB_END_TIME', str(int(time.time() + 1)))
    config = LightningTrainerConfig(
        max_epochs=10,
        min_epochs=0,
        accelerator='cpu',
        devices=1,
        enable_progress_bar=False,
        job_time_share=0.5,
    )
    with pytest.raises(TrainingPaused, match='resubmit'):
        _fit(
            tmp_path=tmp_path,
            config=config,
            model=_ToyModel(learning_rate=0.05, seconds_per_batch=0.1),
        )
    assert not (tmp_path / 'checkpoints' / 'training.json').exists()

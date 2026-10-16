import dataclasses
import logging
from pathlib import Path

import lightning
from lightning import LightningModule
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
import pytest

from tslab.lightning.callbacks import (
    AccumulatingTimerCallback,
    build_callbacks,
    resolve_early_stopping_patience,
)
from tslab.lightning.config import LightningTrainerConfig


@pytest.mark.parametrize(
    ('changes', 'expected'),
    [
        ({}, 250),
        ({'check_val_every_n_epoch': 5}, 50),
        ({'max_epochs': 5}, 1),
        ({'early_stopping_patience': 7}, 7),
    ],
)
def test_patience(*, changes: dict[str, int], expected: int) -> None:
    config = dataclasses.replace(LightningTrainerConfig(), **changes)
    assert resolve_early_stopping_patience(config=config) == expected


def test_callbacks_with_validation(tmp_path: Path) -> None:
    callbacks = build_callbacks(
        config=LightningTrainerConfig(),
        checkpoint_directory=tmp_path,
        has_validation=True,
        has_logger=True,
    )
    assert [type(callback) for callback in callbacks] == [
        ModelCheckpoint,
        EarlyStopping,
        LearningRateMonitor,
        AccumulatingTimerCallback,
    ]
    checkpoint = callbacks[0]
    assert isinstance(checkpoint, ModelCheckpoint)
    assert checkpoint.monitor == 'val_loss_epoch'
    assert checkpoint.filename == '{epoch}-{val_loss_epoch:.2f}'
    assert checkpoint.save_top_k == 3
    assert checkpoint.save_last is True
    early_stopping = callbacks[1]
    assert isinstance(early_stopping, EarlyStopping)
    assert early_stopping.patience == 250
    # Lightning stores min_delta with a sign that depends on the mode.
    assert abs(early_stopping.min_delta) == pytest.approx(0.001)


def test_callbacks_without_validation_or_logger(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        callbacks = build_callbacks(
            config=LightningTrainerConfig(),
            checkpoint_directory=tmp_path,
            has_validation=False,
            has_logger=False,
        )
    assert [type(callback) for callback in callbacks] == [
        ModelCheckpoint,
        AccumulatingTimerCallback,
    ]
    checkpoint = callbacks[0]
    assert isinstance(checkpoint, ModelCheckpoint)
    assert checkpoint.monitor is None
    assert checkpoint.save_top_k == 1
    assert 'no validation loader' in caplog.text


def test_timer_state_survives_a_checkpoint() -> None:
    trainer = lightning.Trainer(logger=False, enable_checkpointing=False, enable_progress_bar=False)
    module = LightningModule()
    timer = AccumulatingTimerCallback()
    timer.sessions_count = 2
    timer.sessions_times = {1: 1.5, 2: 2.0}
    saved: dict[str, object] = {}
    timer.on_save_checkpoint(trainer, module, saved)
    restored = AccumulatingTimerCallback()
    restored.on_load_checkpoint(trainer, module, saved)
    assert restored.sessions_count == 2
    assert restored.sessions_times == {1: 1.5, 2: 2.0}
    assert restored.total_seconds == pytest.approx(3.5)
    restored.on_train_start(trainer, module)
    assert restored.sessions_count == 3


def test_timer_ignores_checkpoints_without_its_state() -> None:
    trainer = lightning.Trainer(logger=False, enable_checkpointing=False, enable_progress_bar=False)
    timer = AccumulatingTimerCallback()
    timer.on_load_checkpoint(trainer, LightningModule(), {'timer_callback_state': 'garbage'})
    assert timer.sessions_count == 0
    assert timer.sessions_times == {}

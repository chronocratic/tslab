import dataclasses
import logging
from pathlib import Path
import pickle
import signal
from threading import Thread
from typing import cast

import lightning
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
import pytest
import torch

from tslab.lab.core.run import JobTimeLimitSignal, Run
from tslab.ml.lightning.callbacks import (
    _SignalStop,
    AccumulatingTimerCallback,
    build_callbacks,
    resolve_early_stopping_patience,
)
from tslab.ml.lightning.config import LightningTrainerConfig
from tslab.ml.lightning.exceptions import TrainingPaused
from tslab.ml.lightning.trainer import _defer_time_limit_signals


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
    assert checkpoint.filename == 'epoch={epoch}-val_loss_epoch={val_loss_epoch:.2f}'
    assert checkpoint.save_top_k == 3
    assert checkpoint.save_last is True
    early_stopping = callbacks[1]
    assert isinstance(early_stopping, EarlyStopping)
    assert early_stopping.patience == 250
    # Lightning stores min_delta with a sign that depends on the mode.
    assert abs(early_stopping.min_delta) == pytest.approx(0.001)


def test_a_monitor_with_a_slash_keeps_checkpoints_in_one_folder(tmp_path: Path) -> None:
    config = LightningTrainerConfig(monitor='val/loss')
    (checkpoint, *_) = build_callbacks(
        config=config, checkpoint_directory=tmp_path, has_validation=True, has_logger=False
    )
    assert isinstance(checkpoint, ModelCheckpoint)
    path = Path(
        checkpoint.format_checkpoint_name(
            {'epoch': torch.tensor(3), 'val/loss': torch.tensor(0.123)}
        )
    )
    assert path.parent == tmp_path.resolve() or path.parent == tmp_path
    assert path.name == 'epoch=3-val_loss=0.12.ckpt'


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


def test_defer_time_limit_signals_is_safe_in_a_worker_thread(tmp_path: Path) -> None:
    run = Run(config=LightningTrainerConfig(), root=tmp_path, relative_path=Path('run'))
    signal_stop = _SignalStop(run=run, checkpoint_directory=tmp_path, deadline=None)
    errors: list[BaseException] = []

    def defer() -> None:
        try:
            with _defer_time_limit_signals(run=run, signal_stop=signal_stop):
                pass
        except ValueError as error:
            errors.append(error)

    thread = Thread(target=defer)
    thread.start()
    thread.join()
    assert errors == []


def test_defer_time_limit_signals_is_safe_without_sigusr2(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delattr(signal, 'SIGUSR2', raising=False)
    run = Run(config=LightningTrainerConfig(), root=tmp_path, relative_path=Path('run'))
    signal_stop = _SignalStop(run=run, checkpoint_directory=tmp_path, deadline=None)
    with _defer_time_limit_signals(run=run, signal_stop=signal_stop):
        pass


def test_signal_stop_callback_can_be_pickled(tmp_path: Path) -> None:
    run = Run(config=LightningTrainerConfig(), root=tmp_path, relative_path=Path('run'))
    callback = _SignalStop(run=run, checkpoint_directory=tmp_path, deadline=None)
    pickle.dumps(callback)


@pytest.mark.parametrize('hook_name', ['on_validation_end', 'on_sanity_check_end'])
def test_signal_stop_saves_at_validation_safe_points(tmp_path: Path, hook_name: str) -> None:
    class Strategy:
        def reduce_boolean_decision(self, *, decision: bool, **_: bool) -> bool:
            return decision

    class FakeTrainer:
        strategy = Strategy()

        def __init__(self) -> None:
            self.saved: list[Path] = []

        def save_checkpoint(self, filepath: Path) -> None:
            self.saved.append(filepath)

    run = Run(config=LightningTrainerConfig(), root=tmp_path, relative_path=Path('run'))
    callback = _SignalStop(
        run=run, checkpoint_directory=tmp_path, signal_received=True, deadline=None
    )
    trainer = FakeTrainer()

    with pytest.raises(JobTimeLimitSignal):
        getattr(callback, hook_name)(trainer, LightningModule())
    assert trainer.saved == [tmp_path / 'last.ckpt']


def test_job_deadline_on_one_rank_stops_all_ranks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    strategy = _FakeStrategy(result=True)
    trainer = _FakeTrainer(strategy=strategy)
    run = Run(config=LightningTrainerConfig(), root=tmp_path, relative_path=Path('run'))
    callback = _SignalStop(run=run, checkpoint_directory=tmp_path, deadline=100.0)
    monkeypatch.setattr('tslab.ml.lightning.callbacks.time.time', lambda: 101.0)

    with pytest.raises(TrainingPaused, match='resubmit'):
        callback.on_train_batch_end(cast('Trainer', trainer), LightningModule(), None, None, 0)

    assert strategy.decisions == [(True, False)]
    assert trainer.saved == [tmp_path / 'last.ckpt']


def test_job_deadline_stops_non_writer_rank(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv('RANK', '1')
    strategy = _FakeStrategy(result=True)
    trainer = _FakeTrainer(strategy=strategy)
    run = Run(config=LightningTrainerConfig(), root=tmp_path, relative_path=Path('run'))
    callback = _SignalStop(run=run, checkpoint_directory=tmp_path, deadline=100.0)
    monkeypatch.setattr('tslab.ml.lightning.callbacks.time.time', lambda: 101.0)

    with pytest.raises(SystemExit) as error:
        callback.on_train_batch_end(cast('Trainer', trainer), LightningModule(), None, None, 0)

    assert error.value.code == 0
    assert strategy.decisions == [(True, False)]
    assert trainer.saved == [tmp_path / 'last.ckpt']


def test_signal_wins_over_job_deadline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    strategy = _FakeStrategy(result=True)
    trainer = _FakeTrainer(strategy=strategy)
    run = Run(config=LightningTrainerConfig(), root=tmp_path, relative_path=Path('run'))
    callback = _SignalStop(
        run=run, checkpoint_directory=tmp_path, signal_received=True, deadline=100.0
    )
    monkeypatch.setattr('tslab.ml.lightning.callbacks.time.time', lambda: 101.0)

    with pytest.raises(JobTimeLimitSignal):
        callback.on_train_batch_end(cast('Trainer', trainer), LightningModule(), None, None, 0)

    assert strategy.decisions == [(True, False)]
    assert trainer.saved == [tmp_path / 'last.ckpt']


def test_signal_stop_suppresses_lightning_sigterm_after_usr2(tmp_path: Path) -> None:
    previous = signal.getsignal(signal.SIGTERM)
    calls: list[int] = []

    def lightning_handler(signal_number: int, frame: object) -> None:
        del frame
        calls.append(signal_number)

    signal.signal(signal.SIGTERM, lightning_handler)
    run = Run(config=LightningTrainerConfig(), root=tmp_path, relative_path=Path('run'))
    callback = _SignalStop(run=run, checkpoint_directory=tmp_path, deadline=None)
    callback.request_stop()
    try:
        callback.on_sanity_check_start(cast('Trainer', object()), LightningModule())
        signal.raise_signal(signal.SIGTERM)
        assert calls == []
    finally:
        signal.signal(signal.SIGTERM, previous)


def test_training_end_saves_last_checkpoint_at_latest_step(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint = ModelCheckpoint(dirpath=tmp_path, save_last=True)
    trainer = lightning.Trainer(
        logger=False, callbacks=[checkpoint], enable_progress_bar=False, max_epochs=1
    )
    last = tmp_path / 'last.ckpt'
    saved: list[Path] = []
    timer = AccumulatingTimerCallback()
    module = LightningModule()

    def save_checkpoint(*, filepath: Path) -> None:
        saved.append(filepath)
        state: dict[str, object] = {'global_step': 42}
        timer.on_save_checkpoint(trainer, module, state)
        torch.save(state, filepath)

    monkeypatch.setattr(trainer, 'save_checkpoint', save_checkpoint)
    timer.sessions_count = 1
    timer.start_time = 0.0
    timer.on_train_end(trainer, module)
    state = torch.load(last, weights_only=False)
    assert saved == [last]
    assert state['global_step'] == 42
    timer_state = state['timer_callback_state']
    assert timer_state['sessions_count'] == 1
    assert timer_state['sessions_times'][1] > 0.0
    assert not list(tmp_path.glob('*.tmp'))


class _FakeStrategy:
    def __init__(self, *, result: bool) -> None:
        self.result = result
        self.decisions: list[tuple[bool, bool]] = []

    def reduce_boolean_decision(self, *, decision: bool, **options: bool) -> bool:
        self.decisions.append((decision, options['all']))
        return self.result


class _FakeTrainer:
    def __init__(self, *, strategy: _FakeStrategy) -> None:
        self.strategy = strategy
        self.saved: list[Path] = []

    def save_checkpoint(self, *, filepath: Path) -> None:
        self.saved.append(filepath)

import dataclasses
from datetime import timedelta
from pathlib import Path
import pickle

import lightning
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, Timer
import pytest
import torch

from tslab.lab.core.config import config_from_yaml, config_to_yaml
from tslab.lab.core.fingerprint import canonicalize, compute_fingerprint
from tslab.lab.core.run import Run
from tslab.ml.lightning.config import LightningTrainerConfig
from tslab.ml.lightning.trainer import LightningTrainer


def fingerprint(*, config: LightningTrainerConfig) -> str:
    return compute_fingerprint(
        document=canonicalize(value=config)  # ty: ignore[invalid-argument-type]
    )


def test_settings_that_cannot_change_weights_are_not_fingerprinted() -> None:
    base = LightningTrainerConfig()
    unchanged = dataclasses.replace(
        base,
        accelerator='cpu',
        devices=2,
        log_every_n_steps=1,
        enable_progress_bar=False,
        checkpoint_save_top_k=1,
        job_time_share=0.9,
    )
    assert fingerprint(config=unchanged) == fingerprint(config=base)


def test_max_training_time_changes_the_fingerprint() -> None:
    base = LightningTrainerConfig()
    changed = dataclasses.replace(base, max_training_time='01:00:00')
    assert fingerprint(config=changed) != fingerprint(config=base)


def test_job_time_share_does_not_change_the_fingerprint() -> None:
    base = LightningTrainerConfig()
    changed = dataclasses.replace(base, job_time_share=0.9)
    assert fingerprint(config=changed) == fingerprint(config=base)


def test_job_time_share_does_not_set_lightning_max_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv('SLURM_JOB_END_TIME', '4102444800')
    config = LightningTrainerConfig(
        max_epochs=1, min_epochs=0, accelerator='auto', devices='auto', enable_progress_bar=False
    )
    trainer = LightningTrainer(config=config)
    run = Run(config=config, root=tmp_path, relative_path=Path('run'))
    checkpoint_directory = tmp_path / 'train'
    built_trainers: list[lightning.Trainer] = []
    build_trainer = trainer._build_trainer  # noqa: SLF001 # why: capture the trainer built by fit

    def capture_build(
        *,
        run: Run,
        checkpoint_directory: Path,
        has_validation: bool,
        max_time: timedelta | None,
        signal_stop: Callback | None = None,
    ) -> lightning.Trainer:
        built = build_trainer(
            run=run,
            checkpoint_directory=checkpoint_directory,
            has_validation=has_validation,
            max_time=max_time,
            signal_stop=signal_stop,
        )
        built_trainers.append(built)
        return built

    def skip_fit(built: lightning.Trainer, **_: object) -> None:
        checkpoint = built.checkpoint_callback
        assert isinstance(checkpoint, ModelCheckpoint)
        last = checkpoint_directory / 'last.ckpt'
        last.touch()
        checkpoint.last_model_path = str(last)
        checkpoint.best_model_path = str(last)

    monkeypatch.setattr(trainer, '_build_trainer', capture_build)
    monkeypatch.setattr(lightning.Trainer, 'fit', skip_fit)
    trainer.fit(
        run=run,
        model=lightning.LightningModule(),
        data=_EmptyData(),
        checkpoint_directory=checkpoint_directory,
    )
    assert len(built_trainers) == 1
    callbacks = getattr(built_trainers[0], 'callbacks')  # noqa: B009 # why: Lightning stores callbacks at runtime
    assert not any(isinstance(callback, Timer) for callback in callbacks)


def test_settings_that_change_weights_are_fingerprinted() -> None:
    base = LightningTrainerConfig()
    changes: list[dict[str, object]] = [
        {'max_epochs': 10},
        {'min_epochs': 0},
        {'check_val_every_n_epoch': 2},
        {'monitor': 'val_accuracy'},
        {'early_stopping_patience': 5},
        {'early_stopping_min_delta': 0.01},
        {'extra_trainer_kwargs': {'gradient_clip_val': 1.0}},
    ]
    for change in changes:
        changed = dataclasses.replace(base, **change)
        assert fingerprint(config=changed) != fingerprint(config=base), change


def test_config_file_round_trip() -> None:
    config = LightningTrainerConfig(
        devices=2,
        max_training_time='1-00:00:00',
        extra_trainer_kwargs={'deterministic': True, 'gradient_clip_val': 0.5, 'precision': '32'},
    )
    text = config_to_yaml(config=config)
    assert config_from_yaml(config_class=LightningTrainerConfig, text=text) == config


def test_extra_trainer_kwargs_cannot_replace_what_tslab_builds() -> None:
    config = LightningTrainerConfig(extra_trainer_kwargs={'logger': False, 'max_time': '00:01:00'})
    with pytest.raises(ValueError, match=r"\['logger', 'max_time'\]"):
        LightningTrainer(config=config)
    LightningTrainer(config=LightningTrainerConfig(extra_trainer_kwargs={'gradient_clip_val': 1.0}))


def test_extra_trainer_kwargs_are_read_only_hashable_and_picklable() -> None:
    config = LightningTrainerConfig(extra_trainer_kwargs={'gradient_clip_val': 1.0})
    with pytest.raises(TypeError):
        config.extra_trainer_kwargs['x'] = 1  # ty: ignore[invalid-assignment]
    assert pickle.loads(pickle.dumps(config)) == config  # noqa: S301 # why: own object
    assert hash(config) == hash(
        LightningTrainerConfig(extra_trainer_kwargs={'gradient_clip_val': 1.0})
    )


class _EmptyData:
    @property
    def info(self) -> object:
        return object()

    def train_loader(self) -> list[torch.Tensor]:
        return []

    def val_loader(self) -> None:
        return None

    def test_loader(self) -> list[torch.Tensor]:
        return []

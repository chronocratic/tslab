import dataclasses

from tslab.config import config_from_yaml, config_to_yaml
from tslab.fingerprint import canonicalize, compute_fingerprint
from tslab.lightning.config import LightningTrainerConfig


def fingerprint(*, config: LightningTrainerConfig) -> str:
    return compute_fingerprint(
        document=canonicalize(value=config)  # ty: ignore[invalid-argument-type]
    )


def test_default_settings() -> None:
    config = LightningTrainerConfig()
    assert (config.max_epochs, config.min_epochs) == (2500, 3)
    assert (config.accelerator, config.devices) == ('auto', 'auto')
    assert config.check_val_every_n_epoch == 1
    assert config.log_every_n_steps == 200
    assert config.monitor == 'val_loss_epoch'
    assert config.checkpoint_save_top_k == 3
    assert config.early_stopping_patience is None
    assert config.early_stopping_min_delta == 0.001
    assert config.max_training_time is None
    assert config.slurm_time_fraction == 0.5
    assert config.extra_trainer_kwargs == {}


def test_settings_that_cannot_change_weights_are_not_fingerprinted() -> None:
    base = LightningTrainerConfig()
    unchanged = dataclasses.replace(
        base,
        accelerator='cpu',
        devices=2,
        log_every_n_steps=1,
        enable_progress_bar=False,
        checkpoint_save_top_k=1,
        max_training_time='01:00:00',
        slurm_time_fraction=0.9,
    )
    assert fingerprint(config=unchanged) == fingerprint(config=base)


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

"""``LightningTrainer``: the ready ``Trainer`` for ``LightningModule`` models."""

from datetime import timedelta
import logging
import os
from pathlib import Path
import pickle
import time
from typing import Any, override, TYPE_CHECKING

import lightning
from lightning import LightningModule
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Timer
import torch

from tslab.artifact import Checkpoint
from tslab.interfaces.data import DataSource
from tslab.interfaces.trainer import Trainer
from tslab.lightning.callbacks import build_callbacks
from tslab.lightning.config import LightningTrainerConfig
from tslab.lightning.data import LightningDataSource
from tslab.lightning.resume import choose_resume_checkpoint
from tslab.lightning.slurm import resolve_training_time
from tslab.paths import require_path
from tslab.tracking.lightning_bridge import lightning_loggers

if TYPE_CHECKING:
    from tslab.run import Run

__all__ = ['LightningTrainer', 'TrainingTimeLimitReached']

_LOGGER = logging.getLogger(__name__)


class TrainingTimeLimitReached(RuntimeError):  # noqa: N818 # why: public API name from the spec
    """Training stopped at its time limit before it finished.

    The checkpoint folder holds the state to resume from: running the same
    command again continues the training.
    """


class LightningTrainer(Trainer[LightningModule]):
    """Trains ``LightningModule`` models with resume, time limits and the run's trackers.

    Example:
        trainer = LightningTrainer(config=LightningTrainerConfig(max_epochs=50))
        checkpoint = trainer.fit(run=run, model=model, data=data, checkpoint_directory=folder)
    """

    def __init__(self, *, config: LightningTrainerConfig | None = None) -> None:
        """Create the trainer.

        Args:
            config: The trainer settings; defaults to ``LightningTrainerConfig()``.
        """
        self._config = config if config is not None else LightningTrainerConfig()

    @property
    def config(self) -> LightningTrainerConfig:
        """Return the trainer settings."""
        return self._config

    def fingerprint_settings(self) -> LightningTrainerConfig:
        """Return the settings; ``NotFingerprinted`` fields are left out of fingerprints."""
        return self._config

    @override
    def fit(
        self,
        *,
        run: 'Run',
        model: LightningModule,
        data: DataSource[object],
        checkpoint_directory: Path,
    ) -> Checkpoint:
        """Train ``model``, resuming from the checkpoints already in ``checkpoint_directory``.

        Raises:
            TrainingTimeLimitReached: If the time limit stopped training before it finished.
        """
        require_path(value=checkpoint_directory, parameter='checkpoint_directory')
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        config = self._config
        max_time = resolve_training_time(
            explicit=config.max_training_time,
            fraction=config.slurm_time_fraction,
            environment=os.environ,
            now=time.time(),
        )
        if max_time is not None and config.min_epochs > 0:
            _LOGGER.warning(
                'a training time limit (%s) is active, but Lightning honours it only after '
                'min_epochs=%d epochs; if one epoch can exceed the limit, pass '
                '--trainer.min-epochs 0',
                max_time,
                config.min_epochs,
            )
        if isinstance(data, LightningDataSource):
            val_loader = None
            has_validation = data.has_validation
        else:
            val_loader = data.val_loader()
            has_validation = val_loader is not None
        trainer = self._build_trainer(
            run=run,
            checkpoint_directory=checkpoint_directory,
            has_validation=has_validation,
            max_time=max_time,
        )
        resume_from = choose_resume_checkpoint(
            directory=checkpoint_directory, monitor=config.monitor
        )
        if isinstance(data, LightningDataSource):
            trainer.fit(model=model, datamodule=data.datamodule, ckpt_path=resume_from)
        else:
            trainer.fit(
                model=model,
                train_dataloaders=data.train_loader(),
                val_dataloaders=val_loader,
                ckpt_path=resume_from,
            )
        if _stopped_by_time_limit(trainer=trainer):
            message = (
                f'stopped by the training time limit ({max_time}) after '
                f'{trainer.current_epoch} epoch(s); resubmit the same command to resume '
                f'from {checkpoint_directory.as_posix()}'
            )
            raise TrainingTimeLimitReached(message)
        return _checkpoint_from(
            trainer=trainer, directory=checkpoint_directory, has_validation=has_validation
        )

    @override
    def load(self, *, model: LightningModule, checkpoint: Checkpoint) -> None:
        """Load the best weights of ``checkpoint`` into ``model``, in place.

        Raises:
            ValueError: If an untrusted checkpoint holds objects that safe loading refuses.
        """
        try:
            state = torch.load(
                f=checkpoint.best_path, map_location='cpu', weights_only=not checkpoint.trusted
            )
        except pickle.UnpicklingError as error:
            if checkpoint.trusted:
                raise
            message = (
                f'{checkpoint.best_path.as_posix()} contains objects that safe loading refuses; '
                'if you trust this file, use Checkpoint.external(path=..., trusted=True)'
            )
            raise ValueError(message) from error
        model.load_state_dict(state_dict=state['state_dict'])

    def _build_trainer(
        self,
        *,
        run: 'Run',
        checkpoint_directory: Path,
        has_validation: bool,
        max_time: timedelta | None,
    ) -> lightning.Trainer:
        config = self._config
        loggers = lightning_loggers(trackers=run.trackers)
        arguments: dict[str, Any] = {
            'max_epochs': config.max_epochs,
            'min_epochs': config.min_epochs,
            'accelerator': config.accelerator,
            'devices': config.devices,
            'check_val_every_n_epoch': config.check_val_every_n_epoch,
            'log_every_n_steps': config.log_every_n_steps,
            'enable_progress_bar': config.enable_progress_bar,
            'max_time': max_time,
            'callbacks': build_callbacks(
                config=config,
                checkpoint_directory=checkpoint_directory,
                has_validation=has_validation,
                has_logger=bool(loggers),
            ),
            'logger': loggers,
            'default_root_dir': checkpoint_directory,
        }
        # Applied last, so that the escape hatch can override TSLab's own arguments.
        arguments.update(config.extra_trainer_kwargs)
        return lightning.Trainer(**arguments)


def _stopped_by_time_limit(*, trainer: lightning.Trainer) -> bool:
    """Return whether the ``Timer`` stopped training before it finished."""
    # Lightning sets `Trainer.callbacks` on the instance and types it as None,
    # so getattr keeps the static checkers from flagging the read.
    callbacks = getattr(trainer, 'callbacks')  # noqa: B009 # why: untyped on the class
    timers = [callback for callback in callbacks if isinstance(callback, Timer)]
    if not timers:
        return False
    remaining = timers[0].time_remaining()
    if remaining is None or remaining > 0:
        return False
    early_stopped = any(
        isinstance(callback, EarlyStopping) and callback.stopped_epoch > 0 for callback in callbacks
    )
    max_epochs = trainer.max_epochs
    finished = max_epochs is not None and 0 <= max_epochs <= trainer.current_epoch
    return not early_stopped and not finished


def _checkpoint_from(
    *, trainer: lightning.Trainer, directory: Path, has_validation: bool
) -> Checkpoint:
    """Name the best and last files that ``ModelCheckpoint`` wrote."""
    callback = trainer.checkpoint_callback
    if not isinstance(callback, ModelCheckpoint) or not callback.last_model_path:
        message = f'training finished without writing a last checkpoint in {directory.as_posix()}'
        raise RuntimeError(message)
    last = Path(callback.last_model_path).name
    best = Path(callback.best_model_path).name if has_validation else ''
    return Checkpoint(directory=directory, best=best or last, last=last)

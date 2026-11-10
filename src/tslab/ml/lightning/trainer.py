"""``LightningTrainer``: the ready ``Trainer`` for ``LightningModule`` models."""

from collections.abc import Generator
from contextlib import contextmanager
from datetime import timedelta
import logging
import os
from pathlib import Path
import pickle
import signal
from threading import current_thread, main_thread
import time
from types import FrameType
from typing import Any, cast, override, Protocol

import lightning
from lightning import LightningModule
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint, Timer
from lightning.pytorch.plugins.environments import SLURMEnvironment
import torch

from tslab.lab.core.files import atomic_write_json
from tslab.lab.core.paths import require_path
from tslab.lab.core.run import JobTimeLimitSignal, Run
from tslab.lab.jobs.slurm import parse_slurm_duration, task_count
from tslab.ml.artifacts import Checkpoint
from tslab.ml.interfaces.data import DataSource
from tslab.ml.interfaces.trainer import Trainer
from tslab.ml.lightning.callbacks import _SignalStop, build_callbacks
from tslab.ml.lightning.config import LightningTrainerConfig
from tslab.ml.lightning.data import LightningDataSource
from tslab.ml.lightning.ranks import agree
from tslab.ml.lightning.resume import choose_resume_checkpoint
from tslab.ml.lightning.slurm import job_training_deadline
from tslab.ml.tracking.lightning_bridge import lightning_loggers

__all__ = ['LightningTrainer']

_LOGGER = logging.getLogger(__name__)
# Arguments that TSLab owns: replacing them would break the trackers, the time limit or resume.
_RESERVED_TRAINER_KWARGS = frozenset(
    {'callbacks', 'logger', 'max_time', 'default_root_dir', 'plugins'}
)


@contextmanager
def _defer_time_limit_signals(*, run: Run, signal_stop: _SignalStop) -> Generator[None, None, None]:
    sigusr2 = getattr(signal, 'SIGUSR2', None)
    if sigusr2 is None or current_thread() is not main_thread():
        yield
        return
    previous_sigusr2 = signal.getsignal(sigusr2)
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def defer_sigusr2(signal_number: int, frame: FrameType | None) -> None:
        signal_stop.request_stop()
        if callable(previous_sigusr2):
            try:
                previous_sigusr2(signal_number, frame)
            except JobTimeLimitSignal:
                return

    def defer_sigterm(signal_number: int, frame: FrameType | None) -> None:
        if signal_stop.signalled or run.time_limit_signalled:
            _LOGGER.info('SIGTERM ignored while the run saves after SIGUSR2')
            return
        if callable(previous_sigterm):
            previous_sigterm(signal_number, frame)
        elif previous_sigterm is not signal.SIG_IGN:
            raise SystemExit(128 + signal_number)

    signal.signal(sigusr2, defer_sigusr2)
    try:
        signal.signal(signal.SIGTERM, defer_sigterm)
        yield
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        signal.signal(sigusr2, previous_sigusr2)


class LightningTrainer(Trainer[LightningModule]):
    """Trains ``LightningModule`` models with resume, time limits and the trackers of the run.

    Example:
        trainer = LightningTrainer(config=LightningTrainerConfig(max_epochs=50))
        checkpoint = trainer.fit(run=run, model=model, data=data, checkpoint_directory=folder)
    """

    def __init__(self, *, config: LightningTrainerConfig | None = None) -> None:
        """Create the trainer.

        Args:
            config: The trainer settings; defaults to ``LightningTrainerConfig()``.

        Raises:
            ValueError: If ``extra_trainer_kwargs`` holds an argument that TSLab sets itself.
        """
        reserved = (
            sorted(_RESERVED_TRAINER_KWARGS & set(config.extra_trainer_kwargs)) if config else []
        )
        if reserved:
            message = (
                f'extra_trainer_kwargs must not set {reserved}: TSLab builds these itself '
                '(use the trainer settings, such as max_training_time, instead)'
            )
            raise ValueError(message)
        self._config = config if config is not None else LightningTrainerConfig()

    @property
    def config(self) -> LightningTrainerConfig:
        """Return the trainer settings."""
        return self._config

    def fingerprint_settings(self) -> LightningTrainerConfig:
        """Return the settings; ``NotFingerprinted`` fields are left out of fingerprints."""
        return self._config

    def agree_across_ranks(self, *, key: str, value: bool) -> bool:
        """Broadcast the cache decision of rank zero to every SLURM task.

        Args:
            key: The stage decision key.
            value: The proposed decision on this rank.

        Returns:
            The decision chosen by rank zero.
        """
        if task_count(environment=os.environ) <= 1:
            return value
        return agree(key=key, value=value, environment=os.environ)

    @override
    def fit(
        self,
        *,
        run: Run,
        model: LightningModule,
        data: DataSource[object],
        checkpoint_directory: Path,
    ) -> Checkpoint:
        """Train ``model``, resuming from the checkpoints already in ``checkpoint_directory``.

        When the training finishes, the checkpoint directory receives a ``training.json``
        that records which rule stopped the training: ``max_epochs``, ``early_stopping``
        or ``max_training_time``.

        Raises:
            TrainingPaused: If the job time share paused training; resubmit to resume.
        """
        require_path(value=checkpoint_directory, parameter='checkpoint_directory')
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        config = self._config
        max_time = (
            None
            if config.max_training_time is None
            else parse_slurm_duration(text=config.max_training_time)
        )
        deadline = job_training_deadline(
            share=config.job_time_share, environment=os.environ, now=time.time()
        )
        if max_time is not None and config.min_epochs > 0:
            _LOGGER.warning(
                'max_training_time (%s) is active, but Lightning honors it only after '
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
        signal_stop = _SignalStop(
            run=run,
            checkpoint_directory=checkpoint_directory,
            signal_received=run.time_limit_signalled,
            deadline=deadline,
        )
        with _defer_time_limit_signals(run=run, signal_stop=signal_stop):
            trainer = self._build_trainer(
                run=run,
                checkpoint_directory=checkpoint_directory,
                has_validation=has_validation,
                max_time=max_time,
                signal_stop=signal_stop,
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
        _write_training_record(trainer=trainer, directory=checkpoint_directory)
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
        if not isinstance(state, dict) or 'state_dict' not in state:
            message = (
                f'{checkpoint.best_path.as_posix()} is not a Lightning checkpoint: '
                "it has no 'state_dict' entry"
            )
            raise ValueError(message)
        model.load_state_dict(state_dict=state['state_dict'])

    def _build_trainer(
        self,
        *,
        run: 'Run',
        checkpoint_directory: Path,
        has_validation: bool,
        max_time: timedelta | None,
        signal_stop: Callback | None = None,
    ) -> lightning.Trainer:
        config = self._config
        loggers = lightning_loggers(trackers=run.trackers)
        callbacks = build_callbacks(
            config=config,
            checkpoint_directory=checkpoint_directory,
            has_validation=has_validation,
            has_logger=bool(loggers),
        )
        if signal_stop is not None:
            callbacks.append(signal_stop)
        arguments: dict[str, Any] = {
            'max_epochs': config.max_epochs,
            'min_epochs': config.min_epochs,
            'accelerator': config.accelerator,
            'devices': config.devices,
            'check_val_every_n_epoch': config.check_val_every_n_epoch,
            'log_every_n_steps': config.log_every_n_steps,
            'enable_progress_bar': config.enable_progress_bar,
            'max_time': max_time,
            'callbacks': callbacks,
            'logger': loggers,
            'default_root_dir': checkpoint_directory,
        }
        if SLURMEnvironment.detect():
            arguments['plugins'] = [SLURMEnvironment(auto_requeue=False)]
        # Applied last, so that the escape hatch can override the other arguments.
        arguments.update(config.extra_trainer_kwargs)
        return lightning.Trainer(**arguments)


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


class _TrainerCallbacks(Protocol):
    """The trainer attributes read after ``fit``: the registered callbacks.

    ``lightning.Trainer`` gets ``callbacks`` from its connector, not from the class
    body, so it has no static type for it; the cast in ``_training_stop_reason`` is
    safe because it only runs after ``fit`` returned normally.
    """

    callbacks: list[Callback]


def _training_stop_reason(*, trainer: lightning.Trainer) -> str:
    """Name the rule that stopped a normal training.

    Args:
        trainer: The trainer after ``fit`` returned normally.

    Returns:
        ``'max_training_time'`` if the training timer ran out, ``'early_stopping'`` if an
        early-stopping callback stopped the training, otherwise ``'max_epochs'``.
    """
    callbacks = cast('_TrainerCallbacks', trainer).callbacks
    if any(
        isinstance(callback, Timer) and _timer_ran_out(timer=callback) for callback in callbacks
    ):
        return 'max_training_time'
    if any(
        isinstance(callback, EarlyStopping) and callback.stopped_epoch > 0 for callback in callbacks
    ):
        return 'early_stopping'
    return 'max_epochs'


def _timer_ran_out(*, timer: Timer) -> bool:
    """Whether the training timer has no time left."""
    remaining = timer.time_remaining()
    return remaining is not None and remaining <= 0


def _write_training_record(*, trainer: lightning.Trainer, directory: Path) -> None:
    """Write ``training.json`` next to the checkpoints of a normal training.

    Args:
        trainer: The trainer after ``fit`` returned normally.
        directory: The checkpoint directory that receives the record.
    """
    stop_reason = _training_stop_reason(trainer=trainer)
    if stop_reason == 'max_training_time':
        _LOGGER.warning(
            'training stopped at max_training_time after %d epochs; the result depends on '
            'hardware speed and is not reproducible from the seed; use max_epochs for '
            'reproducible results',
            trainer.current_epoch,
        )
    atomic_write_json(
        path=directory / 'training.json',
        document={
            'schema': 'tslab.training/1',
            'stop_reason': stop_reason,
            'epochs': trainer.current_epoch,
            'steps': trainer.global_step,
        },
    )

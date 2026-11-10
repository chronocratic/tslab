"""Training callbacks: checkpoints, early stopping, learning rate and total training time."""

import logging
from pathlib import Path
import signal
from threading import current_thread, main_thread
import time
from types import FrameType
from typing import Any, override

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from tslab.lab.core.run import JobTimeLimitSignal, Run
from tslab.ml.lightning.config import LightningTrainerConfig
from tslab.ml.lightning.exceptions import TrainingPaused

__all__ = [
    'AccumulatingTimerCallback',
    'build_callbacks',
    'file_safe_name',
    'resolve_early_stopping_patience',
]

_LOGGER = logging.getLogger(__name__)
_STATE_KEY = 'timer_callback_state'


class _SignalStop(Callback):
    def __init__(
        self,
        *,
        run: Run,
        checkpoint_directory: Path,
        deadline: float | None,
        signal_received: bool = False,
    ) -> None:
        super().__init__()
        self._writer = run.writes_files
        self._signal_received = signal_received
        self._checkpoint_directory = checkpoint_directory
        self._deadline = deadline
        self._sigterm_filter_installed = False

    @property
    def signalled(self) -> bool:
        return self._signal_received

    def request_stop(self) -> None:
        self._signal_received = True

    @override
    def on_sanity_check_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del trainer, pl_module
        self._install_sigterm_filter()

    @override
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del trainer, pl_module
        self._install_sigterm_filter()

    def _install_sigterm_filter(self) -> None:
        if (
            self._sigterm_filter_installed
            or getattr(signal, 'SIGUSR2', None) is None
            or current_thread() is not main_thread()
        ):
            return
        previous = signal.getsignal(signal.SIGTERM)

        def defer_after_usr2(signal_number: int, frame: FrameType | None) -> None:
            if self.signalled:
                _LOGGER.info('SIGTERM ignored while the run saves after SIGUSR2')
                return
            if callable(previous):
                previous(signal_number, frame)
            elif previous is not signal.SIG_IGN:
                raise SystemExit(128 + signal_number)

        signal.signal(signal.SIGTERM, defer_after_usr2)
        self._sigterm_filter_installed = True

    @override
    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        del pl_module, outputs, batch, batch_idx
        self._stop_at_safe_point(trainer=trainer, check_deadline=True)

    @override
    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        self._stop_at_safe_point(trainer=trainer)

    @override
    def on_sanity_check_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        self._stop_at_safe_point(trainer=trainer)

    def _stop_at_safe_point(self, *, trainer: Trainer, check_deadline: bool = False) -> None:
        paused = check_deadline and self._deadline is not None and time.time() >= self._deadline
        stop = trainer.strategy.reduce_boolean_decision(
            decision=self.signalled or paused, all=False
        )
        if not stop:
            return
        _save_last(trainer=trainer, path=self._checkpoint_directory / 'last.ckpt')
        if not self._writer:
            raise SystemExit(0)
        raise self._pause_error()

    def _pause_error(self) -> JobTimeLimitSignal | TrainingPaused:
        """The exception that ends a paused training on the process that writes the result."""
        if self.signalled:
            signal_message = 'SIGUSR2 before the SLURM time limit; resubmit to resume'
            return JobTimeLimitSignal(signal_message)
        message = (
            'training paused at the job_time_share deadline; resubmit to resume from '
            f'{self._checkpoint_directory.as_posix()}'
        )
        return TrainingPaused(message)


class AccumulatingTimerCallback(Callback):
    """Counts training sessions and their durations, across resumes.

    Each ``fit`` call is one session. The state is stored in every checkpoint,
    so the total training time of a model trained over several SLURM jobs is
    still known.
    """

    def __init__(self) -> None:
        """Create a timer with no sessions."""
        super().__init__()
        self.start_time: float | None = None
        self.sessions_count = 0
        self.sessions_times: dict[int, float] = {}

    @property
    def total_seconds(self) -> float:
        """Return the training time summed over all sessions."""
        return sum(self.sessions_times.values())

    @override
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.sessions_count += 1
        self.start_time = time.perf_counter()

    @override
    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._record_session()
        self._save_last_checkpoint(trainer=trainer)
        _LOGGER.info(
            'trained for %d session(s), %.2f s in total; per session: %s',
            self.sessions_count,
            self.total_seconds,
            self.sessions_times,
        )

    @override
    def on_save_checkpoint(
        self, trainer: Trainer, pl_module: LightningModule, checkpoint: dict[str, object]
    ) -> None:
        self._record_session()
        checkpoint[_STATE_KEY] = self._state()

    @override
    def on_load_checkpoint(
        self, trainer: Trainer, pl_module: LightningModule, checkpoint: dict[str, object]
    ) -> None:
        state = checkpoint.get(_STATE_KEY)
        if not isinstance(state, dict):
            return
        count = state.get('sessions_count')
        times = state.get('sessions_times')
        self.sessions_count = count if isinstance(count, int) else 0
        self.sessions_times = (
            {int(session): float(seconds) for session, seconds in times.items()}
            if isinstance(times, dict)
            else {}
        )
        # on_train_start starts the next session.
        self.start_time = None

    def _state(self) -> dict[str, object]:
        return {'sessions_count': self.sessions_count, 'sessions_times': dict(self.sessions_times)}

    def _save_last_checkpoint(self, *, trainer: Trainer) -> None:
        """Write the final model state to ``last.ckpt``, including a mid-epoch stop.

        Every rank must call this: saving ends in a barrier. Only rank zero writes the file.
        """
        callback = trainer.checkpoint_callback
        if not isinstance(callback, ModelCheckpoint):
            return
        if callback.last_model_path:
            path = Path(callback.last_model_path)
        elif callback.dirpath is not None:
            path = Path(callback.dirpath) / 'last.ckpt'
        else:
            return
        _save_last(trainer=trainer, path=path)

    def _record_session(self) -> None:
        if self.start_time is not None:
            self.sessions_times[self.sessions_count] = time.perf_counter() - self.start_time


def _save_last(*, trainer: Trainer, path: Path) -> None:
    """Save ``path`` as the last checkpoint, recorded in the ``ModelCheckpoint`` state.

    The path goes into the saved state, so a resumed run overwrites this file instead
    of writing ``last-v1.ckpt`` next to a stale ``last.ckpt``. Every rank must call this:
    saving ends in a barrier.
    """
    callback = trainer.checkpoint_callback
    if isinstance(callback, ModelCheckpoint):
        callback.last_model_path = str(path)
    trainer.save_checkpoint(filepath=path)


def file_safe_name(*, name: str) -> str:
    """Return ``name`` as it appears in checkpoint file names (``/`` becomes ``_``)."""
    return name.replace('/', '_')


def resolve_early_stopping_patience(*, config: LightningTrainerConfig) -> int:
    """Return the early-stopping patience, in validation checks.

    Args:
        config: The trainer settings.

    Returns:
        ``early_stopping_patience`` if set; otherwise about a tenth of the epoch
        budget, divided by the validation interval so that validating less often
        does not multiply the patience in epochs. Never below 1.
    """
    if config.early_stopping_patience is not None:
        return config.early_stopping_patience
    return max(1, config.max_epochs // (10 * max(1, config.check_val_every_n_epoch)))


def build_callbacks(
    *,
    config: LightningTrainerConfig,
    checkpoint_directory: Path,
    has_validation: bool,
    has_logger: bool,
) -> list[Callback]:
    """Build the callbacks of one training.

    Args:
        config: The trainer settings.
        checkpoint_directory: Folder for the checkpoint files.
        has_validation: Whether the data has a validation loader.
        has_logger: Whether the Lightning trainer gets at least one logger.

    Returns:
        The callbacks, in registration order.
    """
    callbacks: list[Callback] = []
    if has_validation:
        callbacks.append(
            ModelCheckpoint(
                dirpath=checkpoint_directory,
                # A "/" in the metric name would put the file in a sub-folder.
                filename=f'epoch={{epoch}}-{file_safe_name(name=config.monitor)}='
                f'{{{config.monitor}:.2f}}',
                auto_insert_metric_name=False,
                monitor=config.monitor,
                mode='min',
                save_top_k=config.checkpoint_save_top_k,
                save_last=True,
                verbose=True,
            )
        )
        callbacks.append(
            EarlyStopping(
                monitor=config.monitor,
                min_delta=config.early_stopping_min_delta,
                patience=resolve_early_stopping_patience(config=config),
                mode='min',
                verbose=True,
            )
        )
    else:
        _LOGGER.warning(
            'the data has no validation loader: no early stopping and no best-checkpoint '
            'selection; the last checkpoint is used as the best one'
        )
        # save_top_k=1: Lightning writes last.ckpt only in a step that also saved a top-k file.
        callbacks.append(
            ModelCheckpoint(dirpath=checkpoint_directory, save_top_k=1, save_last=True)
        )
    if has_logger:
        callbacks.append(LearningRateMonitor(logging_interval='step'))
    callbacks.append(AccumulatingTimerCallback())
    return callbacks

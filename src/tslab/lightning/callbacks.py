"""Training callbacks: checkpoints, early stopping, learning rate and total training time."""

import logging
from pathlib import Path
import time
from typing import override

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from tslab.lightning.config import LightningTrainerConfig

__all__ = [
    'AccumulatingTimerCallback',
    'build_callbacks',
    'file_safe_name',
    'resolve_early_stopping_patience',
]

_LOGGER = logging.getLogger(__name__)
_STATE_KEY = 'timer_callback_state'


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
        checkpoint[_STATE_KEY] = {
            'sessions_count': self.sessions_count,
            'sessions_times': dict(self.sessions_times),
        }

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

    def _record_session(self) -> None:
        if self.start_time is not None:
            self.sessions_times[self.sessions_count] = time.perf_counter() - self.start_time


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

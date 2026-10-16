"""Trackers that wrap a Lightning logger, and the bridge that hands those loggers to Lightning.

Every ready TSLab tracker wraps a Lightning logger. One experiment
run therefore appears as one W&B, MLflow, TensorBoard or CSV run, which holds
both Lightning's per-batch training metrics and TSLab's stage metrics.
"""

from abc import abstractmethod
from collections.abc import Iterable, Mapping
from pathlib import Path
import shutil
from typing import override

from lightning.pytorch.loggers import Logger

from tslab.interfaces.tracker import RunStatus, Tracker

__all__ = ['LightningLoggerTracker', 'lightning_loggers']


class LightningLoggerTracker(Tracker):
    """A ``Tracker`` backed by a Lightning logger created in ``start``.

    If ``_create_logger`` returns ``None`` (for example W&B could not start),
    every method becomes a no-op: this is the Null Object pattern, so callers
    never check whether the tracker works.
    """

    def __init__(self) -> None:
        """Create an unstarted tracker."""
        self._logger: Logger | None = None
        self._files_directory: Path | None = None

    @property
    def lightning_logger(self) -> Logger | None:
        """Return the wrapped Lightning logger, or ``None`` before start or after a failure."""
        return self._logger

    @abstractmethod
    def _create_logger(self, *, run_directory: Path, run_name: str) -> Logger | None:
        """Build the Lightning logger, or return ``None`` to disable this tracker."""

    @override
    def start(self, *, run_directory: Path, run_name: str) -> None:
        self._files_directory = run_directory / 'trackers' / 'files'
        self._logger = self._create_logger(run_directory=run_directory, run_name=run_name)

    @override
    def log_params(self, *, params: Mapping[str, object]) -> None:
        if self._logger is not None:
            self._logger.log_hyperparams(params=dict(params))

    @override
    def log_metrics(self, *, metrics: Mapping[str, float], step: int | None = None) -> None:
        if self._logger is not None:
            self._logger.log_metrics(metrics=dict(metrics), step=step)
            self._logger.save()

    @override
    def log_file(self, *, path: Path) -> None:
        """Copy the file to ``trackers/files/`` (trackers with remote storage override this)."""
        if self._files_directory is not None:
            self._files_directory.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, self._files_directory / path.name)

    @override
    def finish(self, *, status: RunStatus) -> None:
        if self._logger is not None:
            self._logger.finalize(status='success' if status is RunStatus.FINISHED else 'failed')


def lightning_loggers(*, trackers: Iterable[Tracker]) -> list[Logger]:
    """Return the working Lightning loggers behind ``trackers``.

    ``LightningTrainer`` passes these to ``lightning.Trainer(logger=...)`` so
    Lightning's training metrics reach the same tracker runs.

    Args:
        trackers: The run's trackers; trackers without a Lightning logger are skipped.

    Returns:
        The started Lightning loggers.
    """
    return [
        tracker.lightning_logger
        for tracker in trackers
        if isinstance(tracker, LightningLoggerTracker) and tracker.lightning_logger is not None
    ]

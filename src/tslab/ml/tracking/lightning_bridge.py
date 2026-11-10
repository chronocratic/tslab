"""Trackers that wrap a Lightning logger, and the bridge that hands those loggers to Lightning.

Every ready TSLab tracker wraps a Lightning logger. One experiment
run therefore appears as one W&B, MLflow, TensorBoard or CSV run, which holds
both the per-batch training metrics of Lightning and the stage metrics of TSLab.
"""

from abc import abstractmethod
from collections.abc import Callable, Iterable, Mapping
import functools
import logging
from pathlib import Path
import shutil
from typing import Any, override

from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities import rank_zero_only

from tslab.core.tracker import RunStatus, Tracker

__all__ = ['LightningLoggerTracker', 'TrackerLogger', 'lightning_loggers']

_LOGGER = logging.getLogger(__name__)
_GUARDED_METHODS = ('log_hyperparams', 'log_metrics', 'save', 'finalize')


def _guarded(*, method: Callable[..., Any], name: str) -> Callable[..., Any]:
    @functools.wraps(method)
    def call(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401 # why: mirrors any logger method
        try:
            return method(*args, **kwargs)
        except Exception as error:  # noqa: BLE001 # why: tracking must never stop a run
            _LOGGER.warning('tracker %s failed and was skipped: %s', name, error)
            return None

    return call


def guard_logger(*, logger: Logger) -> Logger:
    """Make the logging methods of ``logger`` warn instead of raise.

    Lightning calls these methods directly, bypassing ``Run``, so an
    unreachable server or a clashing parameter would otherwise end training.
    The methods are replaced on the instance, so ``isinstance`` checks inside
    Lightning still see the original logger class.

    Args:
        logger: The Lightning logger to protect.

    Returns:
        The same logger.
    """
    for name in _GUARDED_METHODS:
        method = getattr(logger, name, None)
        if method is not None:
            setattr(logger, name, _guarded(method=method, name=f'{type(logger).__name__}.{name}'))
    return logger


class LightningLoggerTracker(Tracker):
    """A ``Tracker`` backed by a Lightning logger created in ``start``.

    If ``_create_logger`` returns ``None`` (for example W&B could not start),
    every method becomes a no-op: this is the Null Object pattern, so callers
    never check whether the tracker works. The logger is guarded: its failures
    become warnings, also when Lightning calls it directly.
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
        logger = self._create_logger(run_directory=run_directory, run_name=run_name)
        self._logger = None if logger is None else guard_logger(logger=logger)

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


class TrackerLogger(Logger):
    """A Lightning logger that forwards to a ``Tracker`` that has no Lightning logger.

    This is how the metrics of Lightning reach the local CSV tracker of every
    run, and any tracker you write yourself. The tracker is started and
    finished by ``Run``, not by this logger.
    """

    def __init__(self, *, tracker: Tracker) -> None:
        """Wrap ``tracker``."""
        super().__init__()
        self._tracker = tracker

    @property
    @override
    def name(self) -> str:
        return type(self._tracker).__name__

    @property
    @override
    def version(self) -> str:
        return '0'

    @override
    @rank_zero_only
    def log_hyperparams(self, params: Any, *args: Any, **kwargs: Any) -> None:
        self._tracker.log_params(params=_as_dict(params=params))

    @override
    @rank_zero_only
    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        self._tracker.log_metrics(metrics=dict(metrics), step=step)


def _as_dict(*, params: Any) -> dict[str, object]:  # noqa: ANN401 # why: Lightning passes dicts or namespaces
    if isinstance(params, Mapping):
        return dict(params)
    return dict(vars(params))


def lightning_loggers(*, trackers: Iterable[Tracker]) -> list[Logger]:
    """Return the working Lightning loggers behind ``trackers``.

    ``LightningTrainer`` passes these to ``lightning.Trainer(logger=...)`` so
    the training metrics of Lightning reach the same tracker runs.

    Args:
        trackers: The trackers of the run. A tracker that wraps a Lightning
            logger contributes it; any other tracker gets a ``TrackerLogger``.
            A wrapper whose logger failed to start is skipped.

    Returns:
        The Lightning loggers, all guarded against failures.
    """
    loggers: list[Logger] = []
    for tracker in trackers:
        if not isinstance(tracker, LightningLoggerTracker):
            loggers.append(guard_logger(logger=TrackerLogger(tracker=tracker)))
        elif tracker.lightning_logger is not None:
            loggers.append(tracker.lightning_logger)
    return loggers

import logging
from pathlib import Path
from typing import Any, override

from lightning.pytorch.loggers import Logger
import pytest

from tslab.interfaces.tracker import RunStatus
from tslab.tracking.lightning_bridge import lightning_loggers, LightningLoggerTracker
from tslab.tracking.mlflow_tracker import MLflowTracker


class UnreachableError(ConnectionError):
    def __init__(self) -> None:
        super().__init__('server unreachable')


class BrokenLogger(Logger):
    @property
    @override
    def name(self) -> str:
        return 'broken'

    @property
    @override
    def version(self) -> str:
        return '0'

    @override
    def log_hyperparams(self, params: Any, *args: Any, **kwargs: Any) -> None:
        raise UnreachableError

    @override
    def log_metrics(self, metrics: Any, step: int | None = None) -> None:
        raise UnreachableError

    @override
    def finalize(self, status: str) -> None:
        raise UnreachableError


class BrokenTracker(LightningLoggerTracker):
    @override
    def _create_logger(self, *, run_directory: Path, run_name: str) -> Logger | None:
        return BrokenLogger()


def test_loggers_handed_to_lightning_never_raise(
    *, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    tracker = BrokenTracker()
    tracker.start(run_directory=tmp_path, run_name='run')
    (logger,) = lightning_loggers(trackers=[tracker])
    assert isinstance(logger, BrokenLogger)
    with caplog.at_level(logging.WARNING):
        logger.log_hyperparams({'a': 1})
        logger.log_metrics({'loss': 1.0}, step=1)
        logger.finalize('success')
    assert 'server unreachable' in caplog.text


def test_an_unreachable_mlflow_server_disables_the_tracker(
    *, tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracker = MLflowTracker(tracking_uri='http://127.0.0.1:9')
    with caplog.at_level(logging.WARNING):
        tracker.start(run_directory=tmp_path, run_name='run')
    assert tracker.lightning_logger is None
    tracker.log_metrics(metrics={'a': 1.0})
    tracker.finish(status=RunStatus.FINISHED)

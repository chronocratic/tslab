from collections.abc import Mapping
import logging
from pathlib import Path
from typing import Any, override

from lightning.pytorch.loggers import Logger
import pytest

from tslab.interfaces.tracker import RunStatus, Tracker
from tslab.tracking import mlflow_tracker
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
    class Unreachable:
        def __init__(self, **_settings: object) -> None:
            pass

        @property
        def run_id(self) -> str:
            raise UnreachableError

    monkeypatch.setattr(mlflow_tracker, 'MLFlowLogger', Unreachable)
    tracker = MLflowTracker()
    with caplog.at_level(logging.WARNING):
        tracker.start(run_directory=tmp_path, run_name='run')
    assert tracker.lightning_logger is None
    assert 'server unreachable' in caplog.text
    tracker.log_metrics(metrics={'a': 1.0})
    tracker.finish(status=RunStatus.FINISHED)


class RecordingTracker(Tracker):
    def __init__(self) -> None:
        self.metrics: list[tuple[dict[str, float], int | None]] = []
        self.params: list[dict[str, object]] = []

    @override
    def start(self, *, run_directory: Path, run_name: str) -> None:
        pass

    @override
    def log_params(self, *, params: Mapping[str, object]) -> None:
        self.params.append(dict(params))

    @override
    def log_metrics(self, *, metrics: Mapping[str, float], step: int | None = None) -> None:
        self.metrics.append((dict(metrics), step))

    @override
    def log_file(self, *, path: Path) -> None:
        pass

    @override
    def finish(self, *, status: RunStatus) -> None:
        pass


def test_a_tracker_without_a_lightning_logger_still_gets_the_metrics() -> None:
    tracker = RecordingTracker()
    (logger,) = lightning_loggers(trackers=[tracker])
    logger.log_hyperparams({'lr': 0.1})
    logger.log_metrics({'loss': 1.0}, step=3)
    logger.finalize('success')
    assert tracker.params == [{'lr': 0.1}]
    assert tracker.metrics == [({'loss': 1.0}, 3)]

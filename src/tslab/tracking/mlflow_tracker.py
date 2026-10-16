"""``MLflowTracker``: an MLflow run per TSLab run. Needs the ``tslab[mlflow]`` extra."""

from collections.abc import Mapping
from pathlib import Path
from typing import override

from lightning.pytorch.loggers import Logger, MLFlowLogger

from tslab.interfaces.tracker import RunStatus
from tslab.tracking.lightning_bridge import LightningLoggerTracker

__all__ = ['MLflowTracker']


class MLflowTracker(LightningLoggerTracker):
    """Wraps Lightning's ``MLFlowLogger``.

    MLflow parameters are immutable per run: logging an existing key with a
    new value raises. The tracker remembers which keys it logged and sends
    only new ones.
    """

    def __init__(self, *, experiment: str = 'tslab', tracking_uri: str | None = None) -> None:
        """Configure the tracker.

        Args:
            experiment: MLflow experiment name.
            tracking_uri: MLflow tracking URI. ``None`` uses a SQLite database
                at ``<run>/trackers/mlflow/mlflow.db``, one per run, which
                avoids SQLite write contention between concurrent HPC jobs.
        """
        super().__init__()
        self._experiment = experiment
        self._tracking_uri = tracking_uri
        self._logged_keys: set[str] = set()

    @override
    def _create_logger(self, *, run_directory: Path, run_name: str) -> Logger | None:
        mlflow_directory = (run_directory / 'trackers' / 'mlflow').resolve()
        mlflow_directory.mkdir(parents=True, exist_ok=True)
        tracking_uri = self._tracking_uri or f'sqlite:///{mlflow_directory.as_posix()}/mlflow.db'
        return MLFlowLogger(
            experiment_name=self._experiment,
            run_name=run_name,
            tracking_uri=tracking_uri,
            artifact_location=(mlflow_directory / 'artifacts').as_uri(),
            synchronous=True,
        )

    @override
    def log_params(self, *, params: Mapping[str, object]) -> None:
        new_params = {key: value for key, value in params.items() if key not in self._logged_keys}
        if new_params:
            super().log_params(params=new_params)
            self._logged_keys.update(new_params)

    @override
    def log_file(self, *, path: Path) -> None:
        if isinstance(self._logger, MLFlowLogger) and self._logger.run_id is not None:
            self._logger.experiment.log_artifact(run_id=self._logger.run_id, local_path=str(path))

    @override
    def finish(self, *, status: RunStatus) -> None:
        # Lightning maps 'success' to MLflow FINISHED and anything else to FAILED.
        super().finish(status=status)

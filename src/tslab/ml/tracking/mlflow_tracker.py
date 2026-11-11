"""``MLflowTracker``: an MLflow run per TSLab run. Needs the ``tslab[mlflow]`` extra."""

from collections.abc import Mapping
import logging
from pathlib import Path
from typing import override

from lightning.pytorch.loggers import Logger, MLFlowLogger

from tslab.ml.tracking.lightning_bridge import LightningLoggerTracker

__all__ = ['MLflowTracker']

_LOGGER = logging.getLogger(__name__)


class MLflowTracker(LightningLoggerTracker):
    """Wraps the Lightning ``MLFlowLogger``.

    MLflow parameters are immutable per run: logging an existing key with a
    new value raises. The tracker remembers which keys it logged and sends
    only new ones; a later, different value for a logged key is dropped with
    a warning. An experiment that already exists keeps its own artifact location.
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
        self._logged_values: dict[str, object] = {}

    @override
    def _create_logger(self, *, run_directory: Path, run_name: str) -> Logger | None:
        mlflow_directory = (run_directory / 'trackers' / 'mlflow').resolve()
        mlflow_directory.mkdir(parents=True, exist_ok=True)
        tracking_uri = self._tracking_uri or f'sqlite:///{mlflow_directory.as_posix()}/mlflow.db'
        # Lightning creates the MLflow run lazily on first use; do it here, where a
        # failure disables the tracker instead of ending training.
        try:
            if self._tracking_uri is not None:
                self._ensure_experiment(tracking_uri=tracking_uri)
            logger = MLFlowLogger(
                experiment_name=self._experiment,
                run_name=run_name,
                tracking_uri=tracking_uri,
                artifact_location=(mlflow_directory / 'artifacts').as_uri(),
                synchronous=True,
            )
            _ = logger.run_id
        except Exception as error:  # noqa: BLE001 # why: any MLflow failure disables the tracker
            _LOGGER.warning('MLflow is unavailable, tracking is disabled: %s', error)
            return None
        return logger

    def _ensure_experiment(self, *, tracking_uri: str) -> None:
        """Create the experiment on a shared URI, tolerating a concurrent creation.

        Why: concurrent runs against one tracking URI lose the
        get-then-create race; the second lookup recovers the experiment the
        first run created.
        """
        # why: mlflow is an optional extra, imported only when a tracking URI is used
        from mlflow.exceptions import MlflowException  # noqa: PLC0415
        from mlflow.tracking.client import MlflowClient  # noqa: PLC0415

        client = MlflowClient(tracking_uri=tracking_uri)
        try:
            if client.get_experiment_by_name(self._experiment) is None:
                client.create_experiment(self._experiment)
        except MlflowException:
            if client.get_experiment_by_name(self._experiment) is None:
                raise

    @override
    def log_params(self, *, params: Mapping[str, object]) -> None:
        for key, value in params.items():
            if key in self._logged_values and self._logged_values[key] != value:
                _LOGGER.warning(
                    'MLflow parameters are immutable: %r keeps its first value, not %r', key, value
                )
        new_params = {key: value for key, value in params.items() if key not in self._logged_values}
        if new_params:
            super().log_params(params=new_params)
            self._logged_values.update(new_params)

    @override
    def log_file(self, *, path: Path) -> None:
        if isinstance(self._logger, MLFlowLogger) and self._logger.run_id is not None:
            self._logger.experiment.log_artifact(run_id=self._logger.run_id, local_path=str(path))

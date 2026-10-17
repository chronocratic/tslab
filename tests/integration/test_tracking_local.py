from dataclasses import dataclass
from pathlib import Path

from mlflow.tracking.client import MlflowClient
import pytest

from tslab.core.run import Run
from tslab.tracking.mlflow_tracker import MLflowTracker
from tslab.tracking.tensorboard_tracker import TensorBoardTracker


@dataclass(frozen=True)
class Config:
    seed: int = 1
    learning_rate: float = 0.1


def test_tensorboard_writes_event_files(tmp_path: Path) -> None:
    with Run(
        config=Config(), root=tmp_path, relative_path=Path('a'), trackers=[TensorBoardTracker()]
    ) as run:
        run.log_metrics(metrics={'loss': 0.5}, step=0)
    assert list((tmp_path / 'a' / 'trackers' / 'tensorboard').glob('events.out.tfevents.*'))


def test_mlflow_records_params_metrics_and_failed_status(tmp_path: Path) -> None:
    tracker = MLflowTracker(experiment='test')

    def failing_script() -> None:
        with Run(
            config=Config(), root=tmp_path, relative_path=Path('a'), trackers=[tracker]
        ) as run:
            run.log_metrics(metrics={'loss': 0.5}, step=0)
            # MLflow parameters are immutable; the tracker must skip keys it already sent.
            tracker.log_params(params={'learning_rate': 0.2, 'new_key': 1})
            message = 'boom'
            raise RuntimeError(message)

    with pytest.raises(RuntimeError, match='boom'):
        failing_script()
    database = (tmp_path / 'a' / 'trackers' / 'mlflow' / 'mlflow.db').resolve()
    client = MlflowClient(tracking_uri=f'sqlite:///{database.as_posix()}')
    experiment = client.get_experiment_by_name('test')
    assert experiment is not None
    (mlflow_run,) = client.search_runs(experiment_ids=[experiment.experiment_id])
    assert mlflow_run.data.params == {'seed': '1', 'learning_rate': '0.1', 'new_key': '1'}
    assert mlflow_run.data.metrics == {'loss': 0.5}
    assert mlflow_run.info.run_name == 'a'
    assert mlflow_run.info.status == 'FAILED'

import csv
from dataclasses import dataclass
from pathlib import Path

from lightning.pytorch.loggers import CSVLogger, Logger

from tslab.core.run import Run
from tslab.tracking.csv_tracker import CSVTracker
from tslab.tracking.lightning_bridge import lightning_loggers


@dataclass(frozen=True)
class Config:
    seed: int = 1
    learning_rate: float = 0.1


def read_csv_rows(*, path: Path) -> list[dict[str, str]]:
    with path.open(encoding='utf-8', newline='') as file:
        return list(csv.DictReader(file))


def test_csv_tracker_is_always_on(tmp_path: Path) -> None:
    with Run(config=Config(), root=tmp_path, relative_path=Path('a')) as run:
        assert isinstance(run.trackers[0], CSVTracker)
        run.log_metrics(metrics={'loss': 0.5}, step=3)
    version = tmp_path / 'a' / 'trackers' / 'csv' / 'version_0'
    assert read_csv_rows(path=version / 'metrics.csv') == [{'loss': '0.5', 'step': '3'}]
    assert 'learning_rate: 0.1' in (version / 'hparams.yaml').read_text(encoding='utf-8')


def test_csv_tracker_keeps_metrics_of_earlier_executions(tmp_path: Path) -> None:
    for loss in (0.5, 0.25):
        with Run(config=Config(), root=tmp_path, relative_path=Path('a')) as run:
            run.log_metrics(metrics={'loss': loss}, step=0)
    csv_directory = tmp_path / 'a' / 'trackers' / 'csv'
    first = read_csv_rows(path=csv_directory / 'version_0' / 'metrics.csv')
    second = read_csv_rows(path=csv_directory / 'version_1' / 'metrics.csv')
    assert first[0]['loss'] == '0.5'
    assert second[0]['loss'] == '0.25'


def test_lightning_loggers_returns_started_loggers(tmp_path: Path) -> None:
    csv_tracker = CSVTracker()
    assert lightning_loggers(trackers=[csv_tracker]) == []
    csv_tracker.start(run_directory=tmp_path, run_name='a')
    loggers: list[Logger] = lightning_loggers(trackers=[csv_tracker])
    assert len(loggers) == 1
    assert isinstance(loggers[0], CSVLogger)

import csv
from pathlib import Path

import pytest
import yaml

from tslab.core.status import RunState
from tslab.core.tracker import LocalTracker, RunStatus


def rows(*, path: Path) -> list[dict[str, str]]:
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


@pytest.mark.parametrize(
    ('state', 'status'),
    [
        (RunState.FINISHED, RunStatus.FINISHED),
        (RunState.FAILED, RunStatus.FAILED),
        (RunState.CRASHED, RunStatus.CRASHED),
    ],
)
def test_run_status_maps_each_final_state(state: RunState, status: RunStatus) -> None:
    assert RunStatus.from_run_state(run_state=state) is status


def test_running_is_not_a_final_status() -> None:
    with pytest.raises(ValueError, match='no final status'):
        RunStatus.from_run_state(run_state=RunState.RUNNING)


def test_local_tracker_writes_metrics_and_params(tmp_path: Path) -> None:
    tracker = LocalTracker()
    tracker.start(run_directory=tmp_path, run_name='a')
    tracker.log_params(params={'lr': 0.1, 'model/depth': 3})
    tracker.log_metrics(metrics={'loss': 0.5}, step=1)
    tracker.log_metrics(metrics={'loss': 0.25}, step=2)
    tracker.log_metrics(metrics={'acc': 0.9, 'loss': 0.2}, step=3)
    tracker.finish(status=RunStatus.FINISHED)
    version = tmp_path / 'trackers' / 'csv' / 'version_0'
    assert rows(path=version / 'metrics.csv') == [
        {'step': '1', 'acc': '', 'loss': '0.5'},
        {'step': '2', 'acc': '', 'loss': '0.25'},
        {'step': '3', 'acc': '0.9', 'loss': '0.2'},
    ]
    assert yaml.safe_load((version / 'hparams.yaml').read_text(encoding='utf-8')) == {
        'lr': 0.1,
        'model/depth': 3,
    }


def test_local_tracker_writes_params_yaml_cannot_represent_as_text(tmp_path: Path) -> None:
    tracker = LocalTracker()
    tracker.start(run_directory=tmp_path, run_name='a')
    tracker.log_params(
        params={'mode': RunStatus.FINISHED, 'root': Path('data'), 'nested': {'kind': Path}}
    )
    hparams = tmp_path / 'trackers' / 'csv' / 'version_0' / 'hparams.yaml'
    assert yaml.safe_load(hparams.read_text(encoding='utf-8')) == {
        'mode': 'finished',
        'root': 'data',
        'nested': {'kind': "<class 'pathlib.Path'>"},
    }


def test_local_tracker_uses_the_next_version_folder(tmp_path: Path) -> None:
    for expected in ('version_0', 'version_1'):
        tracker = LocalTracker()
        tracker.start(run_directory=tmp_path, run_name='a')
        tracker.log_metrics(metrics={'x': 1.0})
        assert (tmp_path / 'trackers' / 'csv' / expected / 'metrics.csv').is_file()


def test_local_tracker_copies_files_and_is_inert_before_start(tmp_path: Path) -> None:
    tracker = LocalTracker()
    tracker.log_metrics(metrics={'x': 1.0})
    tracker.log_params(params={'a': 1})
    source = tmp_path / 'note.txt'
    source.write_text('hi', encoding='utf-8')
    tracker.log_file(path=source)
    assert not (tmp_path / 'trackers').exists()
    tracker.start(run_directory=tmp_path, run_name='a')
    tracker.log_file(path=source)
    assert (tmp_path / 'trackers' / 'files' / 'note.txt').read_text(encoding='utf-8') == 'hi'

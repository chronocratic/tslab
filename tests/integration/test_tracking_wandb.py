import contextlib
from dataclasses import dataclass
import logging
import os
from pathlib import Path

from lightning.pytorch.loggers import CSVLogger, WandbLogger
import pytest

from tslab.core.run import Run
from tslab.interfaces.tracker import RunStatus
from tslab.tracking.wandb import tracker as wandb_tracker_module
from tslab.tracking.wandb.log import read_exit_codes, read_records
from tslab.tracking.wandb.tracker import WandbTracker


@dataclass(frozen=True, kw_only=True)
class Config:
    seed: int = 1
    learning_rate: float = 0.1


@pytest.fixture(autouse=True)
def quiet_wandb(monkeypatch: pytest.MonkeyPatch) -> None:
    # Keep W&B local and quiet, whatever the developer's environment says.
    monkeypatch.setenv('WANDB_SILENT', 'true')
    monkeypatch.delenv('WANDB_MODE', raising=False)
    monkeypatch.delenv('WANDB_DISABLE_CODE', raising=False)


def find_transaction_log(*, run_directory: Path) -> Path:
    (log,) = (run_directory / 'trackers' / 'wandb').glob('offline-run-*/run-*.wandb')
    return log


@pytest.mark.slow
@pytest.mark.parametrize(('fail', 'expected_codes'), [(False, [0]), (True, [1])])
def test_wandb_offline_run_records_exit_code(
    tmp_path: Path, *, fail: bool, expected_codes: list[int]
) -> None:
    shell_log = tmp_path / 'slurm.out'
    shell_log.write_text('job output\n', encoding='utf-8')
    tracker = WandbTracker(project='p', mode='offline', shell_log_path=shell_log)

    def script() -> None:
        with Run(
            config=Config(), root=tmp_path, relative_path=Path('a'), trackers=[tracker]
        ) as run:
            run.log_metrics(metrics={'loss': 0.5}, step=0)
            if fail:
                message = 'boom'
                raise RuntimeError(message)

    with pytest.raises(RuntimeError) if fail else contextlib.nullcontext():
        script()
    log = find_transaction_log(run_directory=tmp_path / 'a')
    assert read_exit_codes(path=log) == expected_codes
    assert (log.parent / 'files' / 'slurm.out').is_file()


@pytest.mark.slow
def test_wandb_tracker_settings_and_tags(tmp_path: Path) -> None:
    tracker = WandbTracker(project='p', mode='offline', tags=('extra',))
    tracker.start(run_directory=tmp_path, run_name='ts2vec/Coffee/seed_42')
    logger = tracker.lightning_logger
    assert isinstance(logger, WandbLogger)
    experiment = logger.experiment
    assert experiment.tags == ('extra', 'ts2vec', 'Coffee', 'seed_42')
    assert experiment.settings.x_file_stream_timeout_seconds == 60
    assert Path(experiment.dir).is_relative_to(tmp_path / 'trackers' / 'wandb')
    # The code-saving switch is set only while W&B starts.
    assert 'WANDB_DISABLE_CODE' not in os.environ
    tracker.finish(status=RunStatus.FINISHED)


def test_wandb_falls_back_to_offline_then_to_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    modes: list[str] = []

    def failing_build(
        self: WandbTracker, *, run_directory: Path, run_name: str, tags: list[str], mode: str
    ) -> CSVLogger:
        modes.append(mode)
        if mode == 'online':
            message = 'no network'
            raise ConnectionError(message)
        return CSVLogger(save_dir=run_directory)

    monkeypatch.setattr(WandbTracker, '_build', failing_build)
    tracker = WandbTracker(project='p', mode='online')
    tracker.start(run_directory=tmp_path, run_name='a')
    assert modes == ['online', 'offline']
    assert tracker.lightning_logger is not None

    def always_failing_build(
        self: WandbTracker, *, run_directory: Path, run_name: str, tags: list[str], mode: str
    ) -> CSVLogger:
        message = 'wandb-core did not start'
        raise RuntimeError(message)

    monkeypatch.setattr(WandbTracker, '_build', always_failing_build)
    with caplog.at_level(logging.WARNING):
        tracker.start(run_directory=tmp_path, run_name='a')
    assert tracker.lightning_logger is None
    assert 'including offline' in caplog.text
    # A disabled tracker accepts every call and does nothing (Null Object).
    tracker.log_metrics(metrics={'loss': 1.0})
    tracker.finish(status=RunStatus.FAILED)


def test_missing_wandb_names_the_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(wandb_tracker_module.importlib.util, 'find_spec', lambda _name: None)
    tracker = WandbTracker(project='p', mode='offline')
    with caplog.at_level(logging.WARNING):
        tracker.start(run_directory=tmp_path, run_name='a')
    assert tracker.lightning_logger is None
    assert 'tslab[wandb]' in caplog.text


@pytest.mark.parametrize('content', [b'', b'not a wandb log at all'])
def test_transaction_log_reader_rejects_other_files(tmp_path: Path, *, content: bytes) -> None:
    path = tmp_path / 'run.wandb'
    path.write_bytes(content)
    with pytest.raises(ValueError, match='not a W&B transaction log'):
        list(read_records(path=path))

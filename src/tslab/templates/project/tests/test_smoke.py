"""Run both scripts on the toy data with one training batch, so a broken setup shows in a minute."""

import dataclasses
import json
from pathlib import Path

import pytest
import tslab
from tslab.adapters.chronocratic.models import ChronocraticEncoder
from tslab.cli import main as tslab_main
from tslab.core.status import read_status
from tslab.lightning import LightningTrainerConfig

from {{package_name}} import classify, forecast, toy_data

ONE_BATCH = LightningTrainerConfig(
    max_epochs=1,
    min_epochs=0,
    accelerator='cpu',
    devices=1,
    enable_progress_bar=False,
    extra_trainer_kwargs={'limit_train_batches': 1, 'limit_val_batches': 1},
)
ENCODER = ChronocraticEncoder(device='cpu', batch_size=64)
CLASSIFY_NO_PRESET = (
    'no-preset',
    'model:tstcc',
    'data:ucr',
    '--data.dataset-folder-path',
    'data/ToySpectra',
    '--data.target-column-name',
    'target',
)
FORECAST_NO_PRESET = (
    'no-preset',
    'model:tstcc',
    'data:ett',
    '--data.dataset-file-path',
    'data/toy_hourly.csv.gz',
    '--data.variant',
    'ETTh1',
)


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A working folder with the toy data in ``data/``, as the presets expect."""
    toy_data.main(directory=tmp_path / 'data')
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.parametrize('args', [*([name] for name in classify.PRESETS), CLASSIFY_NO_PRESET])
def test_classify(project: Path, args: tuple[str, ...]) -> None:
    config = tslab.parse_config(
        config_class=classify.ClassifyConfig, presets=classify.PRESETS, args=list(args)
    )
    classify.main(config=dataclasses.replace(config, trainer=ONE_BATCH, encoder=ENCODER))
    snapshot = read_status(path=project / 'runs' / config.run_path / 'status.json')
    assert snapshot is not None
    states = {record.state for record in snapshot.instances.values()}
    assert states == {tslab.InstanceState.DONE}


def test_example_study_runs_one_run_locally() -> None:
    """The example study submits and runs one slice locally, through the CLI.

    Runs from the project root (where ``uv run pytest`` is meant to run), so
    the study's ``uv run`` command finds the venv; ``machines/local.toml`` and
    ``data/`` come from the project, as in the README flow.
    """
    root = Path.cwd()
    toy_data.main(directory=root / 'data')
    # one epoch of the toy data is a single batch, so a CPU run stays short
    extra = [
        '--trainer.max-epochs',
        '1',
        '--trainer.min-epochs',
        '0',
        '--trainer.accelerator',
        'cpu',
        '--trainer.devices',
        '1',
        '--encoder.device',
        'cpu',
    ]
    with pytest.raises(SystemExit) as raised:
        tslab_main(
            args=[
                'study',
                'submit',
                '{{package_name}}.studies:example',
                '--machine',
                'machines/local.toml',
                '--grid',
                'classification',
                '--only',
                'seed=0',
                '--',
                *extra,
            ]
        )
    assert raised.value.code == 0
    with pytest.raises(SystemExit) as raised:
        tslab_main(
            args=['study', 'status', '{{package_name}}.studies:example', '--machine', 'machines/local.toml']
        )
    assert raised.value.code == 0
    status_file = (
        root
        / 'runs'
        / 'example'
        / 'classification'
        / 'ts2vec-archive-split'
        / 'ToySpectra'
        / '0'
        / 'status.json'
    )
    assert json.loads(status_file.read_text(encoding='utf-8'))['run_state'] == 'finished'


@pytest.mark.parametrize('args', [*([name] for name in forecast.PRESETS), FORECAST_NO_PRESET])
def test_forecast(project: Path, args: tuple[str, ...]) -> None:
    config = tslab.parse_config(
        config_class=forecast.ForecastConfig, presets=forecast.PRESETS, args=list(args)
    )
    small = dataclasses.replace(
        config, trainer=ONE_BATCH, encoder=ENCODER, horizons=(24,), window_length=24
    )
    forecast.main(config=small)
    snapshot = read_status(path=project / 'runs' / config.run_path / 'status.json')
    assert snapshot is not None
    assert 'evaluate[h24]' in snapshot.instances
    assert {record.state for record in snapshot.instances.values()} == {tslab.InstanceState.DONE}

"""Run both scripts on the toy data with one training batch, so a broken setup shows in a minute."""

import dataclasses
import json
from pathlib import Path

import pytest
import tslab.lab
from tslab.lab.cli import main as tslab_main
from tslab.lab.core.status import read_status
import tslab.ml
from tslab.ml.lightning import LightningTrainerConfig
import tslab.ts
from tslab.ts.adapters.chronocratic.models import ChronocraticEncoder

from {{package_name}} import classify, forecast, toy_data

STUDY = 'example'
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


@pytest.mark.parametrize('study_name', ['example', 'quick', 'chain'])
def test_study_names_dry_run(
    study_name: str, project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path(__file__).parents[1]
    monkeypatch.setenv('UV_PROJECT', str(root))
    machine = str(root / 'machines' / 'local.toml')
    (project / 'tslab.toml').write_text(
        (root / 'tslab.toml').read_text(encoding='utf-8'), encoding='utf-8'
    )

    with pytest.raises(SystemExit) as raised:
        tslab_main(args=['study', 'submit', study_name, '--machine', machine, '--dry-run'])

    assert raised.value.code == 0


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A working folder with the toy data in ``data/``, as the presets expect."""
    toy_data.main(directory=tmp_path / 'data')
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.parametrize('args', [*([name] for name in classify.PRESETS), CLASSIFY_NO_PRESET])
def test_classify(project: Path, args: tuple[str, ...]) -> None:
    config = tslab.lab.parse_config(
        config_class=classify.ClassifyConfig, presets=classify.PRESETS, args=list(args)
    )
    classify.main(config=dataclasses.replace(config, trainer=ONE_BATCH, encoder=ENCODER))
    snapshot = read_status(path=project / 'runs' / config.run_path / 'status.json')
    assert snapshot is not None
    states = {record.state for record in snapshot.instances.values()}
    assert states == {tslab.lab.InstanceState.DONE}


def test_example_study_runs_one_run_locally(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The example study submits and runs one slice locally, through the CLI.

    Runs in a temporary folder with the toy data; ``UV_PROJECT`` lets the
    ``uv run`` command of the study find the venv of the project from there.
    """
    root = Path(__file__).parents[1]
    monkeypatch.setenv('UV_PROJECT', str(root))
    machine = str(root / 'machines' / 'local.toml')
    # The working folder holds the toy data; copy the project settings there
    # so the short study name expands against the studies package.
    (project / 'tslab.toml').write_text(
        (root / 'tslab.toml').read_text(encoding='utf-8'), encoding='utf-8'
    )
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
    submit = ['study', 'submit', STUDY, '--machine', machine, '--grid', 'classification']
    with pytest.raises(SystemExit) as raised:
        tslab_main(args=[*submit, '--only', 'seed=0', '--', *extra])
    assert raised.value.code == 0
    with pytest.raises(SystemExit) as raised:
        tslab_main(args=['study', 'status', STUDY, '--machine', machine])
    assert raised.value.code == 0
    status_file = (
        project
        / 'runs'
        / 'example'
        / 'classification'
        / 'ts2vec-archive-split'
        / 'ToySpectra'
        / 'seed_0'
        / 'status.json'
    )
    assert json.loads(status_file.read_text(encoding='utf-8'))['run_state'] == 'finished'


@pytest.mark.parametrize('args', [*([name] for name in forecast.PRESETS), FORECAST_NO_PRESET])
def test_forecast(project: Path, args: tuple[str, ...]) -> None:
    config = tslab.lab.parse_config(
        config_class=forecast.ForecastConfig, presets=forecast.PRESETS, args=list(args)
    )
    small = dataclasses.replace(
        config, trainer=ONE_BATCH, encoder=ENCODER, horizons=(24,), window_length=24
    )
    forecast.main(config=small)
    snapshot = read_status(path=project / 'runs' / config.run_path / 'status.json')
    assert snapshot is not None
    assert 'evaluate[h24]' in snapshot.instances
    assert {record.state for record in snapshot.instances.values()} == {
        tslab.lab.InstanceState.DONE
    }


def test_chain_study_runs_locally(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The probe grid runs after the pretrain grid and finds its checkpoint."""
    root = Path(__file__).parents[1]
    monkeypatch.setenv('UV_PROJECT', str(root))
    machine = str(root / 'machines' / 'local.toml')
    (project / 'tslab.toml').write_text(
        (root / 'tslab.toml').read_text(encoding='utf-8'), encoding='utf-8'
    )
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
    submit = ['study', 'submit', 'chain', '--machine', machine, '--only', 'head=svm,ridge']
    with pytest.raises(SystemExit) as raised:
        tslab_main(args=[*submit, '--', *extra])
    assert raised.value.code == 0
    runs = project / 'runs' / 'chain'
    relatives = (
        'pretrain/ts2vec-archive-split/ToySpectra/seed_0',
        'probe/ts2vec-archive-split/ToySpectra/seed_0/svm',
        'probe/ts2vec-archive-split/ToySpectra/seed_0/ridge',
    )
    for relative in relatives:
        folder = runs / relative
        status = json.loads((folder / 'status.json').read_text(encoding='utf-8'))
        assert status['run_state'] == 'finished'
        if relative.endswith('/svm'):
            metrics = json.loads(
                (folder / 'evaluate' / 'default' / 'data.json').read_text(encoding='utf-8')
            )
            assert {'accuracy', 'precision', 'recall', 'f1'} <= metrics.keys()
        if relative.endswith('/ridge'):
            metrics = json.loads(
                (folder / 'evaluate' / 'default' / 'data.json').read_text(encoding='utf-8')
            )
            assert 'ridge_accuracy' in metrics

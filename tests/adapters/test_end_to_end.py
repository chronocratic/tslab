from collections.abc import Generator
import contextlib
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from synthetic_data import CLASSIFICATION_NAME, DATA_DIRECTORY, FORECASTING_FILE

import tslab
from tslab.adapters.chronocratic.datasets import (
    build_chronocratic_data_source,
    ChronocraticClassificationData,
    ChronocraticDataChoice,
    ChronocraticForecastingData,
)
from tslab.adapters.chronocratic.models import (
    build_chronocratic_model,
    ChronocraticEncoder,
    ChronocraticModelChoice,
    MODELS,
)
from tslab.lightning import LightningTrainer, LightningTrainerConfig
from tslab.probes import RidgeProbe, SVMProbe
from tslab.tasks import forecasting_representations

pytestmark = [pytest.mark.chronocratic, pytest.mark.slow]

ENCODER = ChronocraticEncoder(device='cpu', batch_size=64)
CLASSIFICATION_ARGS = (
    'data:ucr',
    '--data.dataset-folder-path',
    str(DATA_DIRECTORY / CLASSIFICATION_NAME),
    '--data.target-column-name',
    'target',
)
FORECASTING_ARGS = (
    'data:ett',
    '--data.dataset-file-path',
    str(DATA_DIRECTORY / FORECASTING_FILE),
    '--data.variant',
    'ETTh1',
)
ONE_BATCH = LightningTrainerConfig(
    max_epochs=1,
    min_epochs=0,
    accelerator='cpu',
    devices=1,
    enable_progress_bar=False,
    extra_trainer_kwargs={'limit_train_batches': 1, 'limit_val_batches': 1},
)
FEW_EPOCHS = LightningTrainerConfig(
    max_epochs=5, min_epochs=0, accelerator='cpu', devices=1, enable_progress_bar=False
)


@dataclass(frozen=True, kw_only=True)
class Config:
    data: ChronocraticDataChoice
    model: ChronocraticModelChoice
    seed: int = 0
    run: tslab.RunOptions = field(default_factory=tslab.RunOptions)


def parse(*args: str) -> Config:
    return tslab.parse_config(config_class=Config, args=list(args))


@contextlib.contextmanager
def run_in(*, root: Path, config: Config, name: str = 'r') -> Generator[tslab.Run]:
    with tslab.Run(config=config, root=root, relative_path=Path(name)) as run:
        yield run


@pytest.mark.parametrize('name', sorted(MODELS))
def test_every_model_trains_and_encodes(tmp_path: Path, name: str) -> None:
    config = parse(*CLASSIFICATION_ARGS, f'model:{name}')
    data = build_chronocratic_data_source(config=config.data)
    assert isinstance(data, ChronocraticClassificationData)
    trainer = LightningTrainer(config=ONE_BATCH)
    with run_in(root=tmp_path, config=config, name=name) as run:
        model = build_chronocratic_model(config=config.model, data_info=data.info)
        checkpoint = tslab.stages.train(run=run, trainer=trainer, model=model, data=data)
        representations = tslab.stages.encode(
            run=run, trainer=trainer, model=model, checkpoint=checkpoint, data=data, encoder=ENCODER
        )
    vectors = representations.tensors['test_representations']
    assert vectors.shape[0] == 28
    assert vectors.ndim == 2


def classify(*, root: Path) -> tslab.Metrics:
    config = parse(*CLASSIFICATION_ARGS, 'model:ts2vec')
    data = build_chronocratic_data_source(config=config.data)
    assert isinstance(data, ChronocraticClassificationData)
    trainer = LightningTrainer(config=FEW_EPOCHS)
    with run_in(root=root, config=config) as run:
        model = build_chronocratic_model(config=config.model, data_info=data.info)
        checkpoint = tslab.stages.train(run=run, trainer=trainer, model=model, data=data)
        representations = tslab.stages.encode(
            run=run, trainer=trainer, model=model, checkpoint=checkpoint, data=data, encoder=ENCODER
        )
        return tslab.stages.evaluate(
            run=run, representations=representations, downstream_model=SVMProbe(n_jobs=1)
        )


def test_ts2vec_classifies_end_to_end_and_reloads(tmp_path: Path) -> None:
    first = classify(root=tmp_path)
    assert first.values['accuracy'] >= 0.75
    meta = tmp_path / 'r' / 'encode' / 'default' / 'meta.json'
    before = meta.stat().st_mtime_ns
    second = classify(root=tmp_path)
    assert second.values == first.values
    assert meta.stat().st_mtime_ns == before


def test_ts2vec_forecasts_end_to_end(tmp_path: Path) -> None:
    config = parse(*FORECASTING_ARGS, 'model:ts2vec', '--model.max-train-length', '512')
    data = build_chronocratic_data_source(config=config.data)
    assert isinstance(data, ChronocraticForecastingData)
    trainer = LightningTrainer(config=FEW_EPOCHS)
    with run_in(root=tmp_path, config=config) as run:
        model = build_chronocratic_model(config=config.model, data_info=data.info)
        checkpoint = tslab.stages.train(run=run, trainer=trainer, model=model, data=data)
        by_horizon = forecasting_representations(
            run=run,
            trainer=trainer,
            model=model,
            checkpoint=checkpoint,
            data=data,
            encoder=ENCODER,
            horizons=(24, 48),
        )
        errors = {
            horizon: tslab.stages.evaluate(
                run=run,
                name=('forecasting', f'h{horizon}'),
                representations=representations,
                downstream_model=RidgeProbe(n_jobs=1),
            ).values['mse_loss']
            for horizon, representations in by_horizon.items()
        }
        states = {key: record.state.value for key, record in run.status().instances.items()}
    assert sorted(errors) == [24, 48]
    assert all(0.0 <= error < 1.0 for error in errors.values())
    assert 'forecast_encode[default]' in states
    assert 'forecast_windows[h24]' not in states

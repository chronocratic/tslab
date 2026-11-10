from dataclasses import dataclass, field
from pathlib import Path

import pytest

from tests.ts.adapters.adapter_helpers import parse, run_in
from tests.ts.adapters.synthetic_data import CLASSIFICATION_NAME, DATA_DIRECTORY, FORECASTING_FILE
import tslab.lab
import tslab.ml
from tslab.ml.lightning import LightningTrainer, LightningTrainerConfig
import tslab.ts
from tslab.ts.adapters.chronocratic.datasets import (
    build_chronocratic_data_source,
    ChronocraticClassificationData,
    ChronocraticDataChoice,
)
from tslab.ts.adapters.chronocratic.models import (
    build_chronocratic_model,
    ChronocraticEncoder,
    ChronocraticModelChoice,
)

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
    run: tslab.lab.RunOptions = field(default_factory=tslab.lab.RunOptions)


@pytest.mark.parametrize('name', ['ts2vec', 'cost', 'tstcc'])
def test_every_model_trains_and_encodes(tmp_path: Path, name: str) -> None:
    config = parse(Config, *CLASSIFICATION_ARGS, f'model:{name}')
    data = build_chronocratic_data_source(config=config.data)
    assert isinstance(data, ChronocraticClassificationData)
    trainer = LightningTrainer(config=ONE_BATCH)
    with run_in(root=tmp_path, config=config, name=name) as run:
        model = build_chronocratic_model(config=config.model, data_info=data.info)
        checkpoint = tslab.ml.stages.train(run=run, trainer=trainer, model=model, data=data)
        representations = tslab.ts.stages.encode(
            run=run, trainer=trainer, model=model, checkpoint=checkpoint, data=data, encoder=ENCODER
        )
    vectors = representations.tensors['test_representations']
    assert vectors.shape[0] == 28
    assert vectors.ndim == 2

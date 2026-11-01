"""Study settings round trip through the real chronocratic choices."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import pytest

import tslab
from tslab.adapters.chronocratic.datasets import ChronocraticDataChoice, data_settings
from tslab.adapters.chronocratic.models import chronocratic_model_choice, model_settings
from tslab.lightning.config import LightningTrainerConfig
from tslab.study import check_paths, from_presets, settings_to_args

pytestmark = pytest.mark.chronocratic

Models = chronocratic_model_choice(augmentations=())


@dataclass(frozen=True, kw_only=True)
class Config:
    """The template classify config shape, on the real choices."""

    model: Models
    data: ChronocraticDataChoice
    trainer: LightningTrainerConfig = field(default_factory=LightningTrainerConfig)
    seed: int = 0


PRESETS: dict[str, tuple[str, object]] = {
    'ts2vec': (
        'ts2vec recipe with two non-default model values',
        Config(
            model=model_settings(choice=Models, name='ts2vec', depth=12, dropout_rate=0.2),
            data=data_settings(
                name='ucr', dataset_folder_path=Path('/d/UCR/Coffee'), target_column_name='target'
            ),
        ),
    ),
    'cost': (
        'cost recipe with a non-default trainer',
        Config(
            model=model_settings(choice=Models, name='cost', learning_rate=1e-4),
            data=data_settings(
                name='ucr', dataset_folder_path=Path('/d/UCR/Coffee'), target_column_name='target'
            ),
            trainer=LightningTrainerConfig(max_epochs=600),
        ),
    ),
    'ett-any': (
        'ETT with the file and variant left to the study',
        Config(
            model=model_settings(choice=Models, name='ts2vec'),
            data=data_settings(
                name='ett', dataset_file_path=tslab.REQUIRED, variant=tslab.REQUIRED
            ),
        ),
    ),
}


def test_recipes_times_datasets_round_trip() -> None:
    recipes = from_presets(presets=PRESETS, names=('ts2vec', 'cost'), without=('data',))
    datasets = {
        'Coffee': {
            'data': 'ucr',
            'data.dataset_folder_path': '/d/UCR/Coffee',
            'data.target_column_name': 'target',
        },
        'BasicMotions': {
            'data': 'uea',
            'data.dataset_folder_path': '/d/UEA/BasicMotions',
            'data.target_column_name': 'target',
        },
        'ETTh1': {
            'data': 'ett',
            'data.dataset_file_path': '/d/ETT/ETTh1.csv',
            'data.variant': 'ETTh1',
        },
    }
    for recipe_name, recipe in recipes.items():
        for dataset in datasets.values():
            settings = {**recipe, **dataset, 'seed': 42}
            assert check_paths(config_class=Config, settings=settings) == []
            config = tslab.parse_config(
                config_class=Config, args=settings_to_args(config_class=Config, settings=settings)
            )
            preset = cast('Config', PRESETS[recipe_name][1])
            assert config.model == preset.model
            assert config.trainer == preset.trainer
            assert config.seed == 42


@pytest.mark.parametrize('name', ['ts2vec', 'cost'])
def test_whole_real_preset_round_trips(name: str) -> None:
    settings = from_presets(presets=PRESETS, names=(name,), without=())[name]
    args = ['no-preset', *settings_to_args(config_class=Config, settings=settings)]
    assert tslab.parse_config(config_class=Config, presets=PRESETS, args=args) == PRESETS[name][1]


def test_required_marker_with_real_classes(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        tslab.parse_config(
            config_class=Config,
            presets=PRESETS,
            args=['ett-any', '--data.dataset-file-path', '/d/ETTh1.csv'],
        )
    assert '--data.variant' in capsys.readouterr().err
    settings = from_presets(presets=PRESETS, names=('ett-any',), without=())['ett-any']
    assert 'data.variant' not in settings

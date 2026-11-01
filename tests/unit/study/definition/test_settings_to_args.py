"""Settings to flags: choice words, flag forms, value forms and the round trip."""

from pathlib import Path
from typing import cast

import pytest

from tests.unit.study.configs import (
    ClassifyConfig,
    CoSTSettings,
    Crop,
    Jitter,
    Models,
    PRESETS,
    SplitMode,
    TS2VecSettings,
    UCRSettings,
)
import tslab
from tslab.core.codec import union_choices
from tslab.study.definition import Settings, settings_to_args


def test_union_choices_reads_subcommand_names() -> None:
    assert union_choices(annotation=Models) == {'ts2vec': TS2VecSettings, 'cost': CoSTSettings}
    assert union_choices(annotation=str | None) is None
    assert union_choices(annotation=int) is None


def test_unnamed_union_member_raises() -> None:
    with pytest.raises(ValueError, match='subcommand'):
        union_choices(annotation=TS2VecSettings | CoSTSettings)


def test_choice_words_and_flags() -> None:
    args = settings_to_args(
        config_class=ClassifyConfig,
        settings={
            'model': 'ts2vec',
            'model.depth': 3,
            'data': 'ucr',
            'data.dataset_folder_path': Path('/d/UCR/Coffee'),
            'seed': 7,
        },
    )
    assert args == [
        'model:ts2vec',
        'data:ucr',
        '--model.depth',
        '3',
        '--data.dataset-folder-path',
        '/d/UCR/Coffee',
        '--seed',
        '7',
    ]


def test_nested_choice_words_come_after_parents() -> None:
    args = settings_to_args(
        config_class=ClassifyConfig,
        settings={
            'model.augmentation': 'crop',
            'model': 'ts2vec',
            'data': 'uea',
            'data.dataset_folder_path': 'x',
        },
    )
    assert args[:3] == ['model:ts2vec', 'data:uea', 'model.augmentation:crop']


def test_bool_values_use_tyro_flag_forms() -> None:
    args = settings_to_args(
        config_class=ClassifyConfig,
        settings={'model': 'ts2vec', 'model.use_mask': False, 'trainer.deterministic': True},
    )
    assert args[1:] == ['--model.no-use-mask', '--trainer.deterministic']


def test_value_forms() -> None:
    args = settings_to_args(
        config_class=ClassifyConfig,
        settings={
            'model': 'ts2vec',
            'model.dims': (),
            'noise_levels': (0.2, 1e-05),
            'data': 'ucr',
            'data.splitting_strategy': SplitMode.MANUAL,
            'data.dataset_folder_path': Path('a b'),
            'trainer.max_training_time': None,
        },
    )
    assert args[2:] == [
        '--model.dims',
        '--noise-levels',
        '0.2',
        '1e-05',
        '--data.splitting-strategy',
        'MANUAL',
        '--data.dataset-folder-path',
        'a b',
        '--trainer.max-training-time',
        'None',
    ]


def test_round_trip_through_parse_config() -> None:
    settings = {
        'model': 'ts2vec',
        'model.depth': 12,
        'model.use_mask': False,
        'model.augmentation': 'crop',
        'model.augmentation.ratio': 0.3,
        'data': 'ucr',
        'data.dataset_folder_path': Path('/d/Coffee'),
        'trainer.max_training_time': '4:00:00',
        'noise_levels': (),
        'seed': 42,
    }
    config = tslab.parse_config(
        config_class=ClassifyConfig,
        args=settings_to_args(config_class=ClassifyConfig, settings=settings),
    )
    assert config.model == TS2VecSettings(depth=12, use_mask=False, augmentation=Crop(ratio=0.3))
    assert config.data == UCRSettings(dataset_folder_path=Path('/d/Coffee'))
    assert config.trainer.max_training_time == '4:00:00'
    assert config.noise_levels == ()
    assert config.seed == 42


def test_default_member_fields_work_when_the_choice_is_not_in_the_settings() -> None:
    # model.augmentation walks the default member Jitter when model is unset.
    config = tslab.parse_config(
        config_class=ClassifyConfig,
        args=settings_to_args(
            config_class=ClassifyConfig,
            settings={
                'model': 'ts2vec',
                'data': 'ucr',
                'data.dataset_folder_path': 'd',
                'model.augmentation.sigma': 0.2,
            },
        ),
    )
    model = config.model
    assert isinstance(model, TS2VecSettings)
    assert isinstance(model.augmentation, Jitter)
    assert model.augmentation.sigma == 0.2


def test_round_trip_with_presets_and_no_preset() -> None:
    args = [
        'no-preset',
        *settings_to_args(
            config_class=ClassifyConfig,
            settings={'model': 'cost', 'data': 'uea', 'data.dataset_folder_path': 'x'},
        ),
    ]
    config = tslab.parse_config(config_class=ClassifyConfig, presets=PRESETS, args=args)
    assert config.model == CoSTSettings()


@pytest.mark.parametrize(
    ('settings', 'match'),
    [
        ({'model': 'ts2vec', 'model.dept': 3}, "no field 'dept'"),
        ({'model': 'tsvec'}, 'ts2vec'),
        ({'model.depth': 3}, 'choose model first'),
        ({'seed': {'a': 1}}, None),
        ({'trainer.max_training_time': 'None'}, 'None'),
    ],
)
def test_bad_settings_raise(settings: dict[str, object], match: str | None) -> None:
    with pytest.raises((ValueError, TypeError), match=match):
        settings_to_args(config_class=ClassifyConfig, settings=cast('Settings', settings))

"""Settings rules found in review: preset fidelity, bad values, bad paths."""

from dataclasses import dataclass, field

import pytest

from tests.unit.study.configs import ClassifyConfig, Models, PRESETS, TS2VecSettings
import tslab
from tslab.study import check_paths, from_presets, settings_to_args


@dataclass(frozen=True, kw_only=True)
class _Train:
    max_epochs: int = 10
    note: str = 'a'


@dataclass(frozen=True, kw_only=True)
class _Config:
    trainer: _Train = field(default_factory=lambda: _Train(max_epochs=100))
    name: str = 'x'
    seed: int = 0


def test_preset_value_equal_to_class_default_is_kept_over_instance_default() -> None:
    presets = {'p': ('keeps the class default', _Config(trainer=_Train(max_epochs=10)))}
    settings = from_presets(presets, names=('p',), without=())['p']
    assert settings == {'trainer.max_epochs': 10}
    args = settings_to_args(config_class=_Config, settings=settings)
    assert tslab.parse_config(config_class=_Config, args=args).trainer.max_epochs == 10


def test_preset_equal_to_the_default_instance_adds_nothing() -> None:
    presets = {'p': ('the default', _Config())}
    assert from_presets(presets, names=('p',), without=()) == {'p': {}}


@dataclass(frozen=True, kw_only=True)
class _ChoiceConfig:
    model: Models = field(default_factory=lambda: TS2VecSettings(depth=3))


def test_choice_default_instance_is_the_base_of_the_diff() -> None:
    presets = {
        'p': ('class default depth', _ChoiceConfig(model=TS2VecSettings(depth=10))),
        'q': ('same as the default instance', _ChoiceConfig()),
    }
    result = from_presets(presets, names=('p', 'q'), without=())
    assert result['p']['model.depth'] == 10
    assert 'model.depth' not in result['q']


def test_none_for_a_non_optional_field_raises() -> None:
    with pytest.raises(ValueError, match='seed'):
        settings_to_args(config_class=ClassifyConfig, settings={'seed': None})
    args = settings_to_args(
        config_class=ClassifyConfig, settings={'trainer.max_training_time': None}
    )
    assert args == ['--trainer.max-training-time', 'None']


def test_walking_through_a_leaf_field_is_a_bad_path() -> None:
    settings = {'trainer.max_epochs.max_epochs': 1}
    bad = check_paths(config_class=ClassifyConfig, settings=settings)
    assert bad == ['trainer.max_epochs.max_epochs']


def test_string_value_starting_with_a_dash_is_one_flag_word() -> None:
    args = settings_to_args(
        config_class=ClassifyConfig, settings={'trainer.max_training_time': '-1:00'}
    )
    assert args == ['--trainer.max-training-time=-1:00']


def test_without_naming_no_field_raises() -> None:
    with pytest.raises(ValueError, match='dta'):
        from_presets(PRESETS, names=('cost-paper',), without=('dta',))

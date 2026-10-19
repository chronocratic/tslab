"""``from_presets``: presets to settings, pinned by round trips."""

from dataclasses import dataclass
from typing import Annotated

import pytest
import tyro

from tests.unit.study.configs import ClassifyConfig, PRESETS, UEASettings
import tslab
from tslab.core.config import union_choices
from tslab.study import from_presets, settings_to_args


@dataclass(frozen=True, kw_only=True)
class _Small:
    size: int = 1


@dataclass(frozen=True, kw_only=True)
class _Optional:
    choice: (
        Annotated[_Small, tyro.conf.subcommand(name='small')]
        | Annotated[None, tyro.conf.subcommand(name='default')]
    ) = None
    seed: int = 0


def test_only_non_default_values() -> None:
    recipes = from_presets(PRESETS, names=('cost-paper',), without=('data',))
    assert recipes == {'cost-paper': {'model': 'cost', 'model.alpha': 0.1}}


def test_names_order_and_unknown_names() -> None:
    assert list(from_presets(PRESETS, names=('cost-paper', 'ts2vec-paper'), without=())) == [
        'cost-paper',
        'ts2vec-paper',
    ]
    with pytest.raises(ValueError, match='ts2vec-paper'):
        from_presets(PRESETS, names=('nope',), without=())


def test_without_drops_the_subtree() -> None:
    settings = from_presets(PRESETS, names=('ts2vec-paper',), without=('data',))['ts2vec-paper']
    assert not any(path == 'data' or path.startswith('data.') for path in settings)


def test_from_presets_skips_required_fields() -> None:
    settings = from_presets(PRESETS, names=('ts2vec-any',), without=())['ts2vec-any']
    assert 'data.dataset_folder_path' not in settings
    assert settings['data.batch_size'] == 4


@pytest.mark.parametrize('name', ['ts2vec-paper', 'cost-paper'])
def test_whole_preset_round_trips(name: str) -> None:
    settings = from_presets(PRESETS, names=(name,), without=())[name]
    args = ['no-preset', *settings_to_args(config_class=ClassifyConfig, settings=settings)]
    assert (
        tslab.parse_config(config_class=ClassifyConfig, presets=PRESETS, args=args)
        == (PRESETS[name][1])
    )


def test_recipe_runs_on_another_data_kind() -> None:
    recipe = from_presets(PRESETS, names=('ts2vec-paper',), without=('data',))['ts2vec-paper']
    settings = {**recipe, 'data': 'uea', 'data.dataset_folder_path': '/d/UEA/BasicMotions'}
    config = tslab.parse_config(
        config_class=ClassifyConfig,
        args=settings_to_args(config_class=ClassifyConfig, settings=settings),
    )
    assert config.model == PRESETS['ts2vec-paper'][1].model
    assert isinstance(config.data, UEASettings)


def test_forgotten_required_field_fails_at_parse() -> None:
    recipe = from_presets(PRESETS, names=('ts2vec-any',), without=())['ts2vec-any']
    with pytest.raises(SystemExit):
        tslab.parse_config(
            config_class=ClassifyConfig,
            args=settings_to_args(config_class=ClassifyConfig, settings=recipe),
        )


def test_union_choices_names_a_none_member() -> None:
    choices = union_choices(annotation=_Optional.__dataclass_fields__['choice'].type)
    assert choices == {'small': _Small, 'default': type(None)}


def test_from_presets_keeps_a_none_choice_as_its_name() -> None:
    presets = {
        'kept': ('A model with the default augmentation', _Optional(choice=None, seed=3)),
        'picked': ('A model with a recipe', _Optional(choice=_Small(size=2), seed=4)),
    }
    assert from_presets(presets, names=('kept', 'picked'), without=()) == {
        'kept': {'choice': 'default', 'seed': 3},
        'picked': {'choice': 'small', 'choice.size': 2, 'seed': 4},
    }

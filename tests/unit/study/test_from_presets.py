"""``from_presets``: presets to settings, pinned by round trips."""

import pytest

from study.configs import ClassifyConfig, PRESETS, UEASettings
import tslab
from tslab.study import from_presets, settings_to_args


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

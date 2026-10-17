import dataclasses
from dataclasses import dataclass
from pathlib import Path
import pickle
import typing
from typing import Any

import chronocratic.models
from chronocratic.models.augmentation import AugmentationProducer
import pytest
from recipes import CropShiftRecipe, JitterThenScaleRecipe

import tslab
from tslab.adapters.chronocratic.models import (
    chronocratic_model_choice,
    ChronocraticModelChoice,
    entry_of,
    model_settings,
    MODELS,
    recipe_subcommand,
)
from tslab.core.fingerprint import canonicalize

pytestmark = pytest.mark.chronocratic

ModelChoice = chronocratic_model_choice(augmentations=(JitterThenScaleRecipe, CropShiftRecipe))
VIEW_PAIR_MODELS = {'tstcc', 'mhccl', 'simclr', 'cost'}


@dataclass(frozen=True)
class Config:
    model: ModelChoice


# why: the choice is a run-time union
def settings_classes(*, choice: Any) -> dict[str, Any]:  # noqa: ANN401
    """Subcommand name to settings class, read back from a choice union."""
    classes: dict[str, Any] = {}
    for member in typing.get_args(choice):
        settings_class, subcommand = typing.get_args(member)[:2]
        classes[subcommand.name] = settings_class
    return classes


# why: generated classes have no static type
def field_names(*, cls: Any) -> set[str]:  # noqa: ANN401
    return {field.name for field in dataclasses.fields(cls)}


def parse(*args: str) -> Config:
    return tslab.parse_config(config_class=Config, args=list(args))


def test_every_parameter_class_is_discovered() -> None:
    expected = {
        name.lower()
        for name in chronocratic.models.__all__
        if dataclasses.is_dataclass(getattr(chronocratic.models, f'{name}ModelParameters', None))
    }
    assert set(MODELS) == expected
    assert len(MODELS) >= 12
    assert {'ts2vec', 'tstcc', 'cost', 'timevae'} <= set(MODELS)


def test_settings_keep_every_field_but_input_dim() -> None:
    classes = settings_classes(choice=ChronocraticModelChoice)
    assert set(classes) == set(MODELS)
    for name, settings_class in classes.items():
        parameters = field_names(cls=MODELS[name].parameters_class)
        expected = parameters - {'input_dim', 'augmentation'}
        assert field_names(cls=settings_class) == expected, name
        assert settings_class.tslab_model == name


def test_defaults_are_chronocratic_defaults() -> None:
    settings = settings_classes(choice=ChronocraticModelChoice)['ts2vec']()
    parameters = MODELS['ts2vec'].parameters_class(input_dim=1)
    for field in dataclasses.fields(settings):
        assert getattr(settings, field.name) == getattr(parameters, field.name), field.name


def test_sequence_length_is_optional() -> None:
    for name in ('cost', 'timevae', 'tst'):
        settings = settings_classes(choice=ChronocraticModelChoice)[name]()
        assert settings.sequence_length is None


def test_recipes_land_only_in_models_of_their_view_type() -> None:
    classes = settings_classes(choice=ModelChoice)
    with_augmentation = {
        name for name, cls in classes.items() if 'augmentation' in field_names(cls=cls)
    }
    assert with_augmentation == VIEW_PAIR_MODELS | {'ts2vec'}
    assert 'augmentation' not in field_names(cls=classes['tst'])


def test_default_keeps_the_models_own_augmentation() -> None:
    assert parse('model:tstcc').model.augmentation is None


def test_recipe_settings_come_from_the_command_line() -> None:
    config = parse(
        'model:tstcc', 'model.augmentation:jitter-then-scale', '--model.augmentation.sigma', '0.5'
    )
    assert config.model.augmentation == JitterThenScaleRecipe(sigma=0.5)


def test_recipe_of_another_view_type_is_rejected_by_the_parser() -> None:
    with pytest.raises(SystemExit):
        parse('model:ts2vec', 'model.augmentation:jitter-then-scale')
    with pytest.raises(SystemExit):
        parse('model:mcl', 'model.augmentation:jitter-then-scale')


def test_model_settings_round_trip_through_a_config_file(tmp_path: Path) -> None:
    path = tmp_path / 'config.yaml'
    for args in (
        ('model:tstcc', 'model.augmentation:jitter-then-scale', '--model.augmentation.ratio', '2'),
        ('model:ts2vec', 'model.augmentation:crop-shift'),
        ('model:cost', '--model.sequence-length', '100'),
        ('model:timenet',),
    ):
        config = parse(*args)
        tslab.save_config(config=config, path=path)
        loaded = tslab.load_config(config_class=Config, path=path)
        assert loaded == config, args
        assert canonicalize(value=loaded) == canonicalize(value=config), args


def test_generated_classes_live_in_the_adapter_module() -> None:
    for settings_class in settings_classes(choice=ModelChoice).values():
        assert settings_class.__module__ == 'tslab.adapters.chronocratic.models.settings'
        assert settings_class.__name__.endswith('Settings')


def test_settings_classes_pickle_round_trip() -> None:
    # why: pickle checks the class against the module attribute, so pickling the
    # latest generation (each call regenerates the classes) is what round-trips
    for settings_class in settings_classes(choice=chronocratic_model_choice()).values():
        instance = settings_class()
        reloaded = pickle.loads(pickle.dumps(instance))  # noqa: S301
        assert type(reloaded) is settings_class
        assert dataclasses.asdict(reloaded) == dataclasses.asdict(instance), settings_class.__name__


def test_recipe_subcommand_names() -> None:
    assert recipe_subcommand(recipe=JitterThenScaleRecipe) == 'jitter-then-scale'
    assert recipe_subcommand(recipe=CropShiftRecipe) == 'crop-shift'


@dataclass(frozen=True)
class UnannotatedRecipe:
    def __call__(self):  # noqa: ANN204 # why: the missing annotation is what this test checks
        return None


@dataclass(frozen=True)
class NoModelRecipe:
    def __call__(self) -> AugmentationProducer[str]:
        raise NotImplementedError


class NotADataclassRecipe:
    def __call__(self) -> AugmentationProducer[str]:
        raise NotImplementedError


@dataclass(frozen=True)
class DefaultRecipe:
    def __call__(self) -> AugmentationProducer[str]:
        raise NotImplementedError


def test_recipe_without_view_type_raises() -> None:
    with pytest.raises(TypeError, match=r'AugmentationProducer\[<view type>\]'):
        chronocratic_model_choice(augmentations=(UnannotatedRecipe,))


def test_recipe_matching_no_model_raises() -> None:
    with pytest.raises(TypeError, match='fits no model'):
        chronocratic_model_choice(augmentations=(NoModelRecipe,))


def test_recipe_that_is_not_a_dataclass_raises() -> None:
    with pytest.raises(TypeError, match='must be a dataclass'):
        chronocratic_model_choice(augmentations=(NotADataclassRecipe,))


def test_recipe_names_must_be_unique_and_not_default() -> None:
    with pytest.raises(ValueError, match='more than once'):
        chronocratic_model_choice(augmentations=(JitterThenScaleRecipe, JitterThenScaleRecipe))
    with pytest.raises(ValueError, match="'default'"):
        chronocratic_model_choice(augmentations=(DefaultRecipe,))


def test_entry_of_rejects_foreign_settings() -> None:
    with pytest.raises(TypeError, match='chronocratic_model_choice'):
        entry_of(settings=JitterThenScaleRecipe())


def test_model_settings_builds_a_member_of_the_choice() -> None:
    settings = model_settings(
        choice=ModelChoice, name='tstcc', augmentation=JitterThenScaleRecipe(sigma=0.5)
    )
    assert entry_of(settings=settings).name == 'tstcc'
    assert settings.augmentation == JitterThenScaleRecipe(sigma=0.5)
    parsed = tslab.parse_config(
        config_class=Config,
        presets={'tstcc-jitter': ('TSTCC with jitter', Config(model=settings))},
        args=['tstcc-jitter'],
    )
    assert parsed.model == settings


def test_model_settings_rejects_unknown_names_and_settings() -> None:
    with pytest.raises(ValueError, match="no model 'ts3vec'"):
        model_settings(choice=ModelChoice, name='ts3vec')
    with pytest.raises(ValueError, match="has no settings \\['max_train_length'\\]"):
        model_settings(choice=ModelChoice, name='tstcc', max_train_length=128)
    with pytest.raises(TypeError, match='chronocratic_model_choice'):
        model_settings(choice=int | str, name='ts2vec')

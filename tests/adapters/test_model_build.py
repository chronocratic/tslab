import dataclasses
import pickle
import typing
from typing import Any

import chronocratic.models
import pytest
from recipes import JitterThenScaleRecipe, RECIPE_CALLS

from tslab.adapters.chronocratic_models import (
    build_chronocratic_model,
    chronocratic_model_choice,
    FingerprintedModel,
    MODELS,
)
from tslab.core.fingerprint import canonicalize
from tslab.tasks import TSClassificationDataInfo, TSForecastingDataInfo

pytestmark = pytest.mark.chronocratic

ModelChoice = chronocratic_model_choice(augmentations=(JitterThenScaleRecipe,))
CLASSIFICATION = TSClassificationDataInfo(num_channels=1, length=64, num_classes=2)
FORECASTING = TSForecastingDataInfo(
    num_channels=8, length=None, target_channels=(7,), covariate_channels=tuple(range(7))
)


# why: generated classes have no static type
def settings(*, name: str, **values: Any) -> Any:  # noqa: ANN401
    for member in typing.get_args(ModelChoice):
        settings_class, subcommand = typing.get_args(member)[:2]
        if subcommand.name == name:
            return settings_class(**values)
    raise KeyError(name)


@pytest.mark.parametrize('name', sorted(MODELS))
def test_every_model_builds_as_its_chronocratic_class(name: str) -> None:
    model = build_chronocratic_model(config=settings(name=name), data_info=CLASSIFICATION)
    chronocratic_class = MODELS[name].model_class
    assert isinstance(model, chronocratic_class)
    assert isinstance(model, FingerprintedModel)
    assert type(model).__name__ == chronocratic_class.__name__
    assert type(model).__module__ == 'tslab.adapters.chronocratic_models.build'


def test_runtime_fields_come_from_the_data() -> None:
    model = build_chronocratic_model(config=settings(name='cost'), data_info=CLASSIFICATION)
    assert model.fingerprint_settings()['runtime'] == {'input_dim': 1, 'sequence_length': 64}
    assert model.hparams['input_dim'] == 1
    assert model.hparams['sequence_length'] == 64


def test_explicit_sequence_length_wins() -> None:
    model = build_chronocratic_model(
        config=settings(name='cost', sequence_length=32), data_info=CLASSIFICATION
    )
    assert model.fingerprint_settings()['runtime'] == {'input_dim': 1, 'sequence_length': 32}


def test_missing_sequence_length_names_the_flag() -> None:
    for name in ('cost', 'timevae', 'tst'):
        with pytest.raises(ValueError, match=r'--model\.sequence-length'):
            build_chronocratic_model(config=settings(name=name), data_info=FORECASTING)


def test_optional_sequence_length_stays_unset_without_a_data_length() -> None:
    model = build_chronocratic_model(config=settings(name='tstcc'), data_info=FORECASTING)
    assert model.fingerprint_settings()['runtime'] == {'input_dim': 8, 'sequence_length': None}


def test_chronocratic_checks_run_at_build() -> None:
    with pytest.raises(ValueError, match='conv_kernel_size must be positive'):
        build_chronocratic_model(
            config=settings(name='simclr', conv_kernel_size=0), data_info=CLASSIFICATION
        )


def test_the_recipe_is_called_once_per_build() -> None:
    RECIPE_CALLS.clear()
    recipe = JitterThenScaleRecipe(sigma=0.5)
    build_chronocratic_model(
        config=settings(name='tstcc', augmentation=recipe), data_info=CLASSIFICATION
    )
    assert RECIPE_CALLS == [recipe]  # noqa: SIM300  # why: calls read before the expected


def test_fingerprint_is_stable_and_resolved() -> None:
    first = build_chronocratic_model(config=settings(name='cost'), data_info=CLASSIFICATION)
    second = build_chronocratic_model(config=settings(name='cost'), data_info=CLASSIFICATION)
    explicit = build_chronocratic_model(
        config=settings(name='cost', sequence_length=64), data_info=CLASSIFICATION
    )
    assert canonicalize(value=first) == canonicalize(value=second) == canonicalize(value=explicit)


def test_fingerprint_follows_the_recipe_settings() -> None:
    def fingerprint(*, sigma: float) -> object:
        recipe = JitterThenScaleRecipe(sigma=sigma)
        model = build_chronocratic_model(
            config=settings(name='tstcc', augmentation=recipe), data_info=CLASSIFICATION
        )
        return canonicalize(value=model)

    assert fingerprint(sigma=0.5) != fingerprint(sigma=0.8)


def test_pickle_keeps_the_class_and_the_fingerprint() -> None:
    model = build_chronocratic_model(
        config=settings(name='tstcc', augmentation=JitterThenScaleRecipe()),
        data_info=CLASSIFICATION,
    )
    # why: round trip of an object made in this test
    copy = pickle.loads(pickle.dumps(model))  # noqa: S301
    assert type(copy) is type(model)
    assert copy.fingerprint_settings() == model.fingerprint_settings()


def test_foreign_settings_are_rejected() -> None:
    with pytest.raises(TypeError, match='chronocratic_model_choice'):
        build_chronocratic_model(config=JitterThenScaleRecipe(), data_info=CLASSIFICATION)


def test_settings_cover_every_parameter_field() -> None:
    model = build_chronocratic_model(
        config=settings(name='tstcc', augmentation=JitterThenScaleRecipe()),
        data_info=CLASSIFICATION,
    )
    recorded = model.fingerprint_settings()
    settings_part, runtime_part = recorded['settings'], recorded['runtime']
    assert isinstance(settings_part, dict)
    assert isinstance(runtime_part, dict)
    parameters = {
        field.name for field in dataclasses.fields(chronocratic.models.TSTCCModelParameters)
    }
    assert set(settings_part) | set(runtime_part) == parameters

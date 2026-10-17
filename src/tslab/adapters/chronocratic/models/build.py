"""Build chronocratic models from TSLab settings, as fingerprintable subclasses."""

from collections.abc import Callable, Mapping
import dataclasses
import sys
from typing import Any

from lightning import LightningModule

from tslab.adapters.chronocratic.models.settings import entry_of, ModelEntry, MODELS
from tslab.tasks import TSDataInfo

__all__ = ['FingerprintedModel', 'build_chronocratic_model']


class FingerprintedModel(LightningModule):
    """A chronocratic model that TSLab can fingerprint.

    ``build_chronocratic_model`` returns a subclass of both this class and the
    chronocratic model class, and records the settings it was built from.
    """

    tslab_build_settings: Mapping[str, object] | None = None

    def fingerprint_settings(self) -> Mapping[str, object]:
        """Return the model name, its settings and the values taken from the data.

        Raises:
            RuntimeError: If the model was not made by ``build_chronocratic_model``.
        """
        if self.tslab_build_settings is None:
            message = 'build this model with build_chronocratic_model to fingerprint it'
            raise RuntimeError(message)
        return self.tslab_build_settings


def _fingerprinted_subclass(*, entry: ModelEntry) -> Callable[..., FingerprintedModel]:
    model_class: Any = entry.model_class
    name = model_class.__name__
    namespace = {'__module__': __name__, '__doc__': model_class.__doc__}
    subclass = type(name, (FingerprintedModel, model_class), namespace)
    subclass.__qualname__ = name
    # Pickle finds classes by module and name.
    setattr(sys.modules[__name__], name, subclass)
    return subclass


_MODEL_CLASSES: Mapping[str, Callable[..., FingerprintedModel]] = {
    name: _fingerprinted_subclass(entry=entry) for name, entry in MODELS.items()
}


# why: a generated or third-party dataclass type
def _requires(*, parameters_class: Any, name: str) -> bool:  # noqa: ANN401
    return any(
        field.name == name
        and field.default is dataclasses.MISSING
        and field.default_factory is dataclasses.MISSING
        for field in dataclasses.fields(parameters_class)
    )


def _runtime_fields(
    *, entry: ModelEntry, values: dict[str, object], data_info: TSDataInfo
) -> dict[str, object]:
    runtime: dict[str, object] = {'input_dim': data_info.num_channels}
    if 'sequence_length' not in values:
        return runtime
    length = values.pop('sequence_length')
    if length is None:
        length = data_info.length
    if length is None and _requires(
        parameters_class=entry.parameters_class, name='sequence_length'
    ):
        message = (
            f'{entry.name} needs the length of its training samples, and this data has no fixed '
            'length; set it with --model.sequence-length (for example 128)'
        )
        raise ValueError(message)
    runtime['sequence_length'] = length
    return runtime


# why: generated settings classes have no static type
def build_chronocratic_model(
    *,
    config: Any,  # noqa: ANN401
    data_info: TSDataInfo,
) -> FingerprintedModel:
    """Build the chronocratic model that ``config`` describes.

    ``input_dim`` comes from the channel count of the data; an unset
    ``sequence_length`` takes the length of the data. The parameter class of
    chronocratic is built first, so its own checks run here. An augmentation
    recipe is called and its producer
    passed to the model.

    Args:
        config: A model chosen through ``chronocratic_model_choice``.
        data_info: Facts about the training data.

    Returns:
        A subclass of the chronocratic model class that also offers
        ``fingerprint_settings()``.

    Raises:
        TypeError: If ``config`` is not chronocratic model settings.
        ValueError: If the model needs a sequence length that neither the
            config nor the data give, or chronocratic rejects a value.
    """
    entry = entry_of(settings=config)
    values: dict[str, Any] = {
        field.name: getattr(config, field.name) for field in dataclasses.fields(config)
    }
    runtime = _runtime_fields(entry=entry, values=values, data_info=data_info)
    recorded = {'model': entry.name, 'settings': dict(values), 'runtime': dict(runtime)}
    recipe = values.pop('augmentation', None)
    if recipe is not None:
        values['augmentation'] = recipe()
    parameters = entry.parameters_class(**values, **runtime)
    model = _MODEL_CLASSES[entry.name](
        **{field.name: getattr(parameters, field.name) for field in dataclasses.fields(parameters)}
    )
    model.tslab_build_settings = recorded
    return model

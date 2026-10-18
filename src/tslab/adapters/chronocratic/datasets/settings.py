"""Command-line settings for every chronocratic datamodule, generated from its constructor."""

from collections.abc import Mapping
import dataclasses
import functools
import inspect
import operator
from pathlib import Path
import re
import typing
from typing import Annotated, Any

import chronocratic.datasets
from lightning import LightningDataModule
import tyro

from tslab.adapters.chronocratic.generated import make_settings_class, unknown_settings
from tslab.core.fingerprint import NotFingerprinted

__all__ = [
    'DATAMODULES',
    'EXCLUDED_PARAMETERS',
    'NOT_FINGERPRINTED_PARAMETERS',
    'ChronocraticDataChoice',
    'DataModuleEntry',
    'data_settings',
    'datamodule_class_of',
]

# TSLab picks the loader per use; only the sliding-window loaders of chronocratic read the rest.
EXCLUDED_PARAMETERS = frozenset(
    {'loader_mode', 'loader_strict_batch_size', 'seq_len', 'forecast_horizon', 'step'}
)
NOT_FINGERPRINTED_PARAMETERS = frozenset({'num_workers'})
_DATAMODULE_ATTRIBUTE = 'tslab_datamodule'
_VARIADIC = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)


@dataclasses.dataclass(frozen=True, kw_only=True)
class DataModuleEntry:
    """One chronocratic datamodule found in ``chronocratic.datasets``.

    Attributes:
        name: Subcommand name (``'ucr'``, ``'ett'``).
        datamodule_class: The chronocratic ``LightningDataModule`` class.
        settings_class: The generated settings dataclass.
    """

    name: str
    datamodule_class: Any
    settings_class: Any


def _subcommand(*, class_name: str) -> str:
    """Return the first word of a class name: ``UCRClassificationDataModule`` gives ``ucr``."""
    first = re.match(r'[A-Z]+(?![a-z])|[A-Z][a-z]+', class_name)
    return (first.group(0) if first else class_name).lower()


# why: a third-party class
def _constructor_hints(*, datamodule_class: Any) -> dict[str, Any]:  # noqa: ANN401
    # chronocratic-datasets imports Path only for type checkers, so it is supplied here.
    namespace = {'Path': Path, **vars(chronocratic.datasets)}
    return typing.get_type_hints(datamodule_class.__init__, localns=namespace)


def _settings_class(*, datamodule_class: Any) -> type:  # noqa: ANN401 # why: a third-party class
    hints = _constructor_hints(datamodule_class=datamodule_class)
    fields: list[tuple[str, Any, Any]] = []
    for parameter in inspect.signature(datamodule_class.__init__).parameters.values():
        if parameter.name == 'self' or parameter.kind in _VARIADIC:
            continue
        if parameter.name in EXCLUDED_PARAMETERS:
            continue
        annotation: Any = hints[parameter.name]
        if parameter.name in NOT_FINGERPRINTED_PARAMETERS:
            annotation = Annotated[annotation, NotFingerprinted]
        default = (
            dataclasses.field()
            if parameter.default is inspect.Parameter.empty
            else dataclasses.field(default=parameter.default)
        )
        fields.append((parameter.name, annotation, default))
    return make_settings_class(
        name=f'{datamodule_class.__name__}Settings',
        fields=fields,
        namespace={_DATAMODULE_ATTRIBUTE: datamodule_class, '__doc__': datamodule_class.__doc__},
        module=__name__,
    )


def _discover_datamodules() -> dict[str, DataModuleEntry]:
    entries: dict[str, DataModuleEntry] = {}
    for class_name in chronocratic.datasets.__all__:
        candidate = getattr(chronocratic.datasets, class_name)
        if (
            isinstance(candidate, type)
            and issubclass(candidate, LightningDataModule)
            and class_name.endswith('DataModule')
            and not class_name.startswith('Base')
            and not inspect.isabstract(candidate)
        ):
            name = _subcommand(class_name=class_name)
            if name in entries:
                message = (
                    f'{class_name} and {entries[name].datamodule_class.__name__} both give '
                    f'the subcommand {name!r}'
                )
                raise ValueError(message)
            entries[name] = DataModuleEntry(
                name=name,
                datamodule_class=candidate,
                settings_class=_settings_class(datamodule_class=candidate),
            )
    return entries


DATAMODULES: Mapping[str, DataModuleEntry] = _discover_datamodules()


# why: the union is built at run time; Any lets it serve as a field annotation
def _data_choice() -> Any:  # noqa: ANN401
    members: list[Any] = []
    for name, entry in DATAMODULES.items():
        settings_class: Any = entry.settings_class
        members.append(Annotated[settings_class, tyro.conf.subcommand(name=name)])
    return functools.reduce(operator.or_, members)


ChronocraticDataChoice = _data_choice()


def datamodule_class_of(*, settings: object) -> Any:  # noqa: ANN401 # why: a third-party class
    """Return the datamodule class a settings object belongs to.

    Raises:
        TypeError: If ``settings`` is not a member of ``ChronocraticDataChoice``.
    """
    datamodule_class = getattr(type(settings), _DATAMODULE_ATTRIBUTE, None)
    if datamodule_class is None:
        message = (
            f'{type(settings).__qualname__} is not chronocratic data settings; '
            'use a dataset chosen through ChronocraticDataChoice'
        )
        raise TypeError(message)
    return datamodule_class


# why: the settings classes are generated at run time
def data_settings(*, name: str, **values: object) -> Any:  # noqa: ANN401
    """Return the settings of one dataset, for writing presets in code.

    Args:
        name: The subcommand name of the dataset, for example ``'ucr'`` or ``'ett'``.
        **values: Settings that differ from the defaults; required ones (such
            as the dataset path) must be given.

    Raises:
        ValueError: If no datamodule has this name, or it has no such setting.
    """
    if name not in DATAMODULES:
        message = f'no dataset {name!r}; choose from {sorted(DATAMODULES)}'
        raise ValueError(message)
    settings_class = DATAMODULES[name].settings_class
    unknown = unknown_settings(settings_class=settings_class, values=values)
    if unknown:
        message = f'dataset {name!r} has no settings {unknown}'
        raise ValueError(message)
    return settings_class(**values)

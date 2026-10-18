"""Saving, loading and parsing run configurations.

Configs are plain frozen dataclasses. This module writes them as plain YAML
and reads them back, strictly and safely:

- **Safe:** files are read with ``yaml.safe_load``, and only types reachable
  from the annotations of the config class are ever constructed. A config file can
  never execute code (unlike pickle or the full loader of PyYAML).
- **Strict:** unknown keys, wrong value types and unknown union members raise
  ``ConfigFileError`` naming the exact field, so a typo in a hand-edited file
  is never silently ignored.
- **Union members are explicit:** a dataclass stored in a union-typed field
  carries a ``__type__`` key with its class name, because two members can have
  identical fields (for example the parameter classes of two models).

Encoding is driven by *values*, decoding by *annotations*. A field typed
``SomeObject | None`` therefore saves fine while its value is ``None``; a
non-``None`` value of an unsupported type raises ``TypeError`` when saving.

Parsing the command line is delegated to tyro; ``parse_config`` only adds
the defaults of TSLab.

``REQUIRED`` marks a preset value its author cannot know (a dataset path, a
cluster folder). A preset that carries it must be given the value on the
command line, or tyro stops with ``Missing ... --<flag>``. A parsed config
never holds the marker, and ``save_config`` raises ``TypeError`` for an
unparsed preset that still holds it.
"""

from collections.abc import Mapping, Sequence
import dataclasses
from enum import Enum
import logging
from pathlib import Path
import types
from typing import (
    Annotated,
    Any,
    get_args,
    get_origin,
    get_type_hints,
    Literal,
    NoReturn,
    TYPE_CHECKING,
    Union,
)

import tyro
import yaml

from tslab.core.paths import atomic_write_text, require_path
from tslab.core.text import qualified_name

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

__all__ = [
    'CONFIG_FORMAT_VERSION',
    'REQUIRED',
    'ConfigFileError',
    'config_from_yaml',
    'config_to_data',
    'config_to_yaml',
    'decode_dataclass',
    'load_config',
    'parse_config',
    'save_config',
    'strip_annotated',
    'union_choices',
    'union_members',
]

_LOGGER = logging.getLogger(__name__)
CONFIG_FORMAT_VERSION = 1

# An optional marker for a preset value the preset author cannot know; the
# user must give it on the command line (tyro stops with "Missing ...").
REQUIRED: Any = tyro.MISSING
_TYPE_KEY = '__type__'
NO_PRESET = 'no-preset'
_SCALARS = (bool, int, float, str)
_MAPPING_ARGUMENT_COUNT = 2


class ConfigFileError(ValueError):
    """Raised when a config file does not match the config class."""


# The type of a ``tyro.conf.subcommand(name=...)`` metadata item, computed from
# the public API so the private ``_confstruct`` module is never imported.
_SUBCOMMAND_TYPE = type(tyro.conf.subcommand(name='x'))


def union_choices(*, annotation: object) -> dict[str, type] | None:
    """The tyro subcommand names of a choice union, or ``None`` for other types.

    A *choice union* is a ``Union`` (or ``|``) with at least one dataclass
    member, optionally wrapped in ``Annotated``: the field that picks a model
    or a dataset. Every member must be named with
    ``tyro.conf.subcommand(name=...)``; the name of a member is the ``name`` of the
    first such metadata item on it.

    Args:
        annotation: A type annotation.

    Returns:
        ``{subcommand name: member class}`` (a named ``None`` member maps to
        ``type(None)``), or ``None`` when the annotation is not a choice union.

    Raises:
        ValueError: If a dataclass member carries no ``tyro.conf.subcommand``
            name, naming the member.
    """
    annotation = strip_annotated(annotation=annotation)
    if get_origin(annotation) not in {Union, types.UnionType}:
        return None
    members = get_args(annotation)
    has_dataclass = any(
        isinstance(member, type) and dataclasses.is_dataclass(member)
        for member in (strip_annotated(annotation=member) for member in members)
    )
    if not has_dataclass:
        return None
    choices: dict[str, type] = {}
    for member in members:
        if get_origin(member) is not Annotated:
            stripped: object = member
            name = None
        else:
            stripped = strip_annotated(annotation=member)
            name = next(
                (meta.name for meta in get_args(member)[1:] if type(meta) is _SUBCOMMAND_TYPE), None
            )
        if stripped is None or stripped is type(None):  # Annotated[None, ...] normalizes
            if name is None:
                continue  # a bare None carries no name
            choices[name] = type(None)
            continue
        if not isinstance(stripped, type) or not dataclasses.is_dataclass(stripped):
            continue
        if name is None:
            message = (
                f'union member {stripped.__qualname__} is unnamed; name it with '
                f'tyro.conf.subcommand(name=...)'
            )
            raise ValueError(message)
        choices[name] = stripped
    return choices


def union_members(*, annotation: object) -> tuple[object, ...] | None:
    """Return the members of a union annotation, or ``None`` for any other annotation.

    Members wrapped in ``Annotated`` are unwrapped, and a bare ``None`` member
    becomes ``type(None)``.

    Args:
        annotation: A type annotation.
    """
    if get_origin(annotation) not in {Union, types.UnionType}:
        return None
    # Subcommand names wrap members in Annotated; a bare None member means NoneType.
    members = (strip_annotated(annotation=member) for member in get_args(annotation))
    return tuple(type(None) if member is None else member for member in members)


def strip_annotated(*, annotation: object) -> object:
    """Return the wrapped type of ``Annotated[T, ...]``, or ``annotation`` itself.

    Args:
        annotation: A type annotation.
    """
    return get_args(annotation)[0] if get_origin(annotation) is Annotated else annotation


def _item_annotations(*, annotation: object, length: int, location: str) -> list[object]:
    """Return one annotation per item of a list or tuple of ``length`` items."""
    arguments = get_args(annotation)
    if not arguments:
        return [Any] * length
    if get_origin(annotation) is tuple and arguments[-1] is not Ellipsis:
        if len(arguments) != length:
            message = f'{location}: expected {len(arguments)} items, got {length}'
            raise ConfigFileError(message)
        return list(arguments)
    return [arguments[0]] * length


def _mapping_value_annotation(*, annotation: object) -> object:
    arguments = get_args(annotation)
    return arguments[1] if len(arguments) == _MAPPING_ARGUMENT_COUNT else Any


def _fail(*, data: object, expected: str, location: str) -> NoReturn:
    message = f'{location}: expected {expected}, got {data!r}'
    raise ConfigFileError(message)


# Encoding (values to YAML-ready data) -------------------------------------------------


def _encode(*, value: object, annotation: object, location: str) -> object:
    annotation = strip_annotated(annotation=annotation)
    if isinstance(value, Enum):  # before scalars: StrEnum and IntEnum members are str and int
        return value.name
    if value is None or isinstance(value, _SCALARS):
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _encode_dataclass(value=value, annotation=annotation, location=location)
    if isinstance(value, list | tuple):
        annotations = _item_annotations(annotation=annotation, length=len(value), location=location)
        return [
            _encode(value=item, annotation=item_annotation, location=f'{location}[{index}]')
            for index, (item, item_annotation) in enumerate(zip(value, annotations, strict=True))
        ]
    if isinstance(value, Mapping):
        value_annotation = _mapping_value_annotation(annotation=annotation)
        for key in value:
            if not isinstance(key, str):
                message = (
                    f'{location}: mapping keys must be str to be saved in a config file, '
                    f'got {key!r}; YAML would load other keys back as text'
                )
                raise TypeError(message)
        return {
            key: _encode(value=item, annotation=value_annotation, location=f'{location}.{key}')
            for key, item in value.items()
        }
    message = (
        f'{location}: cannot save a value of type {type(value).__qualname__} in a config file; '
        f'configs may hold only dataclasses, scalars, Path, Enum, tuples, lists, dicts and None'
    )
    raise TypeError(message)


def _encode_dataclass(
    *, value: 'DataclassInstance', annotation: object, location: str
) -> dict[str, object]:
    encoded: dict[str, object] = {}
    members = union_members(annotation=annotation)
    if members is not None:
        choices = [member for member in members if member is not type(None)]
        if len(choices) > 1:
            _reject_duplicate_names(members=choices, location=location)
            encoded[_TYPE_KEY] = type(value).__name__
    elif annotation is not Any and annotation is not type(value):
        message = (
            f'{location}: value of type {type(value).__qualname__} is stored in a field '
            f'annotated {annotation!r}; annotate the field with the exact class or a union'
        )
        raise TypeError(message)
    hints = get_type_hints(type(value), include_extras=True)
    for field in dataclasses.fields(value):
        if field.init:
            encoded[field.name] = _encode(
                value=getattr(value, field.name),
                annotation=hints[field.name],
                location=f'{location}.{field.name}',
            )
    return encoded


def _reject_duplicate_names(*, members: Sequence[object], location: str) -> None:
    """Raise if two dataclass members of a union share a class name.

    ``__type__`` holds only the class name, so such members could not be told apart.
    """
    names = [
        member.__name__
        for member in members
        if isinstance(member, type) and dataclasses.is_dataclass(member)
    ]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        message = (
            f'{location}: union members share the class name {duplicates}; '
            f'give each member a distinct class name so {_TYPE_KEY} can tell them apart'
        )
        raise TypeError(message)


# Decoding (YAML data to values, driven by annotations) --------------------------------


def _decode(*, data: object, annotation: object, location: str) -> object:
    annotation = strip_annotated(annotation=annotation)
    if annotation is Any:
        return data
    members = union_members(annotation=annotation)
    if members is not None:
        return _decode_union(data=data, members=members, location=location)
    origin = get_origin(annotation)
    if origin is Literal:
        if data in get_args(annotation):
            return data
        _fail(data=data, expected=f'one of {list(get_args(annotation))}', location=location)
    if origin in {tuple, list, Sequence}:
        return _decode_sequence(data=data, annotation=annotation, location=location)
    if origin in {dict, Mapping}:
        return _decode_mapping(data=data, annotation=annotation, location=location)
    if isinstance(annotation, type):
        return _decode_class(data=data, cls=annotation, location=location)
    message = f'{location}: config files do not support fields annotated {annotation!r}'
    raise ConfigFileError(message)


def _decode_sequence(*, data: object, annotation: object, location: str) -> object:
    if not isinstance(data, list):
        _fail(data=data, expected='list', location=location)
    annotations = _item_annotations(annotation=annotation, length=len(data), location=location)
    items = [
        _decode(data=item, annotation=item_annotation, location=f'{location}[{index}]')
        for index, (item, item_annotation) in enumerate(zip(data, annotations, strict=True))
    ]
    return items if get_origin(annotation) is list else tuple(items)


def _decode_mapping(*, data: object, annotation: object, location: str) -> dict[str, object]:
    if not isinstance(data, dict):
        _fail(data=data, expected='mapping', location=location)
    value_annotation = _mapping_value_annotation(annotation=annotation)
    decoded: dict[str, object] = {}
    for key, item in data.items():
        if not isinstance(key, str):
            _fail(data=key, expected='str key', location=location)
        decoded[key] = _decode(data=item, annotation=value_annotation, location=f'{location}.{key}')
    return decoded


def _decode_class(*, data: object, cls: type, location: str) -> object:
    if cls is type(None):
        if data is None:
            return None
        _fail(data=data, expected='null', location=location)
    if issubclass(cls, Enum):
        if isinstance(data, str) and data in cls.__members__:
            return cls[data]
        _fail(data=data, expected=f'one of {list(cls.__members__)}', location=location)
    if issubclass(cls, Path):
        if isinstance(data, str):
            return cls(data)
        _fail(data=data, expected='path string', location=location)
    if dataclasses.is_dataclass(cls):
        return decode_dataclass(data=data, cls=cls, location=location)
    if cls in _SCALARS:
        return _decode_scalar(data=data, cls=cls, location=location)
    message = f'{location}: config files do not support fields of type {cls.__qualname__}'
    raise ConfigFileError(message)


def _decode_scalar(*, data: object, cls: type, location: str) -> object:
    is_number = isinstance(data, int | float) and not isinstance(data, bool)
    if cls is bool and isinstance(data, bool):
        return data
    if cls is int and is_number and isinstance(data, int):
        return data
    if cls is float and is_number and isinstance(data, int | float):
        return float(data)
    if cls is str and isinstance(data, str):
        return data
    _fail(data=data, expected=cls.__name__, location=location)


def _decode_union(*, data: object, members: tuple[object, ...], location: str) -> object:
    if data is None and type(None) in members:
        return None
    candidates = [member for member in members if member is not type(None)]
    dataclass_members = [
        member
        for member in candidates
        if isinstance(member, type) and dataclasses.is_dataclass(member)
    ]
    if isinstance(data, dict) and _TYPE_KEY in data:
        name = data[_TYPE_KEY]
        matches = [member for member in dataclass_members if member.__name__ == name]
        if len(matches) > 1:
            message = f'{location}: {_TYPE_KEY} {name!r} names several union members'
            raise ConfigFileError(message)
        if len(matches) != 1:
            names = [member.__name__ for member in dataclass_members]
            message = f'{location}: {_TYPE_KEY} {name!r} is not one of {names}'
            raise ConfigFileError(message)
        body = {key: value for key, value in data.items() if key != _TYPE_KEY}
        return decode_dataclass(data=body, cls=matches[0], location=location)
    if isinstance(data, dict) and len(dataclass_members) > 1:
        names = [member.__name__ for member in dataclass_members]
        message = f'{location}: missing {_TYPE_KEY}; it must name one of {names}'
        raise ConfigFileError(message)
    errors: list[str] = []
    for member in candidates:
        try:
            return _decode(data=data, annotation=member, location=location)
        except ConfigFileError as error:
            errors.append(str(error))
    message = f'{location}: {data!r} matches no member of the union: ' + '; '.join(errors)
    raise ConfigFileError(message)


def decode_dataclass[C](*, data: object, cls: type[C], location: str) -> C:
    """Build a dataclass instance from plain data, strictly, driven by its annotations.

    This is the decoder behind ``load_config``. It is public so other layers
    (for example machine profiles read from TOML) validate with the same rules
    and report errors in the same form.

    Args:
        data: A mapping of field names to plain values.
        cls: The dataclass to build.
        location: Dotted path of ``data`` for error messages, for example ``'config'``.

    Returns:
        The instance.

    Raises:
        ConfigFileError: If a key is unknown, a required field is missing, a
            value has the wrong type, or the constructor rejects the values.
            The message names the field.
        TypeError: If ``cls`` is not a dataclass.
    """
    if not dataclasses.is_dataclass(cls):
        message = f'{location}: {cls.__qualname__} is not a dataclass'
        raise TypeError(message)
    if not isinstance(data, dict):
        _fail(data=data, expected=f'mapping for {cls.__name__}', location=location)
    init_fields = [field for field in dataclasses.fields(cls) if field.init]
    fields = {field.name for field in init_fields}
    unknown = sorted(set(data) - fields)
    if unknown:
        message = f'{location}: unknown keys {unknown} for {cls.__name__}; known: {sorted(fields)}'
        raise ConfigFileError(message)
    missing = [
        field.name
        for field in init_fields
        if field.name not in data
        and field.default is dataclasses.MISSING
        and field.default_factory is dataclasses.MISSING
    ]
    if missing:
        names = ', '.join(f'{location}.{name}' for name in missing)
        message = (
            f'{location}: missing required {"field" if len(missing) == 1 else "fields"} {names}'
        )
        raise ConfigFileError(message)
    hints = get_type_hints(cls, include_extras=True)
    values = {
        name: _decode(data=item, annotation=hints[name], location=f'{location}.{name}')
        for name, item in data.items()
    }
    try:
        return cls(**values)
    except (TypeError, ValueError) as error:
        message = f'{location}: {cls.__name__} rejected the values: {error}'
        raise ConfigFileError(message) from error


def config_to_data(*, config: object) -> dict[str, object]:
    """Convert a config dataclass to plain YAML data: the ``config`` section of ``config.yaml``.

    ``Run`` also sends this data to the trackers, so numbers stay numbers there.

    Args:
        config: A dataclass instance.

    Returns:
        Nested dicts, lists, strings, numbers, booleans and ``None``.

    Raises:
        TypeError: If ``config`` is not a dataclass instance or holds a value
            that config files cannot represent.
    """
    if not dataclasses.is_dataclass(config) or isinstance(config, type):
        message = f'config must be a dataclass instance, got {type(config).__name__}'
        raise TypeError(message)
    return _encode_dataclass(value=config, annotation=type(config), location='config')


def config_to_yaml(*, config: object) -> str:
    """Serialize a config dataclass to YAML.

    Args:
        config: A dataclass instance.

    Returns:
        YAML text with ``format_version``, ``config_type`` and ``config`` keys.

    Raises:
        TypeError: If ``config`` is not a dataclass instance or holds a value
            that config files cannot represent.
    """
    document = {
        'format_version': CONFIG_FORMAT_VERSION,
        'config_type': qualified_name(cls=type(config)),
        'config': config_to_data(config=config),
    }
    return yaml.safe_dump(document, sort_keys=False, allow_unicode=True)


def config_from_yaml[C](*, config_class: type[C], text: str) -> C:
    """Rebuild a config from YAML written by ``config_to_yaml``.

    Args:
        config_class: The dataclass to build.
        text: YAML text.

    Returns:
        The config instance.

    Raises:
        ConfigFileError: If the text does not match ``config_class``.
    """
    document = yaml.safe_load(text)
    if not isinstance(document, dict) or document.get('format_version') != CONFIG_FORMAT_VERSION:
        message = f'not a TSLab config file (format_version {CONFIG_FORMAT_VERSION} expected)'
        raise ConfigFileError(message)
    if document.get('config_type') != qualified_name(cls=config_class):
        _LOGGER.warning(
            'config file was written for %s, loading it as %s',
            document.get('config_type'),
            qualified_name(cls=config_class),
        )
    config = decode_dataclass(data=document.get('config'), cls=config_class, location='config')
    if not isinstance(config, config_class):
        message = f'decoded config is {type(config).__qualname__}, not {config_class.__qualname__}'
        raise ConfigFileError(message)
    return config


def save_config(*, config: object, path: Path) -> None:
    """Write ``config`` to ``path`` atomically (see ``config_to_yaml``).

    Args:
        config: A dataclass instance.
        path: Destination, usually ``<run folder>/config.yaml``.
    """
    require_path(value=path, parameter='path')
    atomic_write_text(path=path, text=config_to_yaml(config=config))


def load_config[C](*, config_class: type[C], path: Path) -> C:
    """Load a config saved by a run, for example to reproduce it.

    Example::

        config = tslab.load_config(config_class=ExperimentConfig, path=Path('runs/a/config.yaml'))
        config = tslab.parse_config(
            config_class=ExperimentConfig, default=config, args=['--seed', '3']
        )

    Args:
        config_class: The dataclass to build.
        path: A ``config.yaml`` written by ``Run``.

    Returns:
        The config instance.
    """
    require_path(value=path, parameter='path')
    return config_from_yaml(config_class=config_class, text=path.read_text(encoding='utf-8'))


def parse_config[C](
    *,
    config_class: type[C],
    presets: Mapping[str, tuple[str, C]] | None = None,
    default: C | None = None,
    args: Sequence[str] | None = None,
) -> C:
    """Build a config from the command line with tyro.

    Flags may appear before or after union subcommands such as ``model:ts2vec``
    (``CascadeSubcommandArgs`` of tyro), so ``model:ts2vec --seed 3`` and
    ``--seed 3 model:ts2vec`` both work.

    Args:
        config_class: The config dataclass.
        presets: Named presets ``{name: (description, config)}``, the
            experiment registry. The first argument then selects a preset or
            ``no-preset``. The union choices of a preset (its model, its dataset)
            are fixed, while flags still override its values; ``no-preset``
            starts from ``config_class`` with every choice open.
        default: Starting values, for example a config loaded with ``load_config``.
            Not allowed together with ``presets``.
        args: Arguments to parse; ``None`` means ``sys.argv[1:]``.

    Returns:
        The parsed config.

    Raises:
        ValueError: If both ``presets`` and ``default`` are given, or a preset
            is named ``no-preset``.
    """
    cascade = (tyro.conf.CascadeSubcommandArgs,)
    if presets is not None:
        if default is not None:
            message = 'pass either presets= or default=, not both'
            raise ValueError(message)
        choice = _preset_choice(config_class=config_class, presets=presets)
        parsed = tyro.cli(choice, args=args, config=cascade)
        # tyro returns Any; every preset is an instance of config_class.
        if not isinstance(parsed, config_class):
            message = f'a preset is not a {config_class.__qualname__}: {type(parsed).__qualname__}'
            raise TypeError(message)
        return parsed
    if default is None:
        return tyro.cli(config_class, args=args, config=cascade)
    return tyro.cli(config_class, default=default, args=args, config=cascade)


type _Presets[C] = Mapping[str, tuple[str, C]]


# why: the return value is a union type built at run time, which no annotation can name
def _preset_choice[C](*, config_class: type[C], presets: _Presets[C]) -> Any:  # noqa: ANN401
    """Return the tyro union of ``no-preset`` and the presets.

    ``AvoidSubcommands`` on a preset turns each union field that has a default
    into the type of that default, so ``model:other`` is rejected instead of
    silently dropping the model settings of the preset.
    """
    if NO_PRESET in presets:
        message = f'{NO_PRESET!r} is reserved for starting without a preset'
        raise ValueError(message)
    free: Any = config_class  # why: ty accepts a class in Annotated only when typed Any
    members: list[Any] = [
        Annotated[
            free, tyro.conf.subcommand(name=NO_PRESET, description='Every setting from the flags.')
        ]
    ]
    for name, (description, config) in presets.items():
        preset_class: Any = type(config)
        members.append(
            Annotated[
                preset_class,
                tyro.conf.subcommand(name=name, default=config, description=description),
                tyro.conf.AvoidSubcommands,
            ]
        )
    return Union[tuple(members)]  # noqa: UP007 # why: built from a runtime tuple

"""Typed dotted settings: the declarative twin of command-line flags.

``Settings`` maps a dotted field path of the script's config class to a value
(``{'model': 'ts2vec', 'model.depth': 10}``). ``settings_to_args`` turns them
into the flags tyro reads back into the same config; the round trip
settings to flags to ``tslab.parse_config`` is pinned by tests.
"""

from collections.abc import Mapping
import dataclasses
from dataclasses import MISSING
from enum import Enum
from pathlib import Path
import types
from typing import Annotated, Any, cast, get_args, get_origin, get_type_hints, Union

from tslab.core.config import union_choices

__all__ = ['SettingValue', 'Settings', 'field_annotation', 'settings_to_args']

type SettingValue = (
    str
    | int
    | float
    | bool
    | Path
    | Enum
    | tuple['SettingValue', ...]
    | list['SettingValue']
    | None
)
type Settings = Mapping[str, SettingValue]


def _strip(*, annotation: object) -> object:
    return get_args(annotation)[0] if get_origin(annotation) is Annotated else annotation


def _flag(*, path: str) -> str:
    return '--' + path.replace('_', '-')


def _bool_flag(*, path: str, value: bool) -> list[str]:
    """The flag form tyro expects for a bool: ``--x`` or ``--parent.no-x``."""
    if value:
        return [_flag(path=path)]
    head, _, last = path.rpartition('.')
    return [_flag(path=(head + '.' if head else '') + 'no-' + last)]


def _value_args(*, value: SettingValue, path: str) -> list[str]:
    if isinstance(value, Enum):  # before str and int: StrEnum and IntEnum are both
        return [value.name]
    if isinstance(value, Path):
        return [value.as_posix()]
    if isinstance(value, tuple | list):
        return [
            argument for item in value for argument in _value_args(value=item, path=f'{path} item')
        ]
    if value is None or isinstance(value, str | int | float):
        return [str(value)]
    message = (
        f'{path}: cannot turn a value of type {type(value).__qualname__} into a flag; '
        f'settings may hold str, int, float, bool, Path, Enum, tuples, lists and None'
    )
    raise TypeError(message)


def _allows_none(*, annotation: object) -> bool:
    annotation = _strip(annotation=annotation)
    if annotation is Any:
        return True
    if get_origin(annotation) in {Union, types.UnionType}:
        return any(member is type(None) for member in get_args(annotation))
    return False


def _default_choice_name(*, holder: type, field_name: str) -> str | None:
    """The subcommand name of a union field's default instance, if it has one."""
    dc_fields = dataclasses.fields(cast('Any', holder))  # a dataclass, checked by the caller
    field = next(item for item in dc_fields if item.name == field_name)
    instance: object = None
    if field.default is not MISSING:
        instance = field.default
    elif field.default_factory is not MISSING:  # type: ignore[misc]
        instance = field.default_factory()  # type: ignore[misc]
    if instance is None:
        return None
    choices = union_choices(annotation=get_type_hints(holder, include_extras=True)[field_name])
    if choices is None:
        return None
    for name, cls in choices.items():
        if type(instance) is cls:
            return name
    return None


def field_annotation(*, config_class: type, path: str, settings: Settings) -> object:
    """The annotation at a dotted path of ``config_class``.

    At a choice union, the member named in the same settings mapping is used;
    an unchosen union uses the union field's default member when it has one.

    Args:
        config_class: The config dataclass the path is relative to.
        path: Dotted field path, e.g. ``model.depth``.
        settings: The settings mapping the path comes from; it names the
            chosen members of the unions on the way down.

    Returns:
        The annotation of the field at the end of the path.

    Raises:
        ValueError: If a class on the way down has no such field, if a chosen
            name is not a member, or if a path walks through an unchosen union
            that has no default member.
    """
    parts = path.split('.')
    holder: object = config_class
    annotation: object = Any
    for position, part in enumerate(parts):
        if not (isinstance(holder, type) and dataclasses.is_dataclass(holder)):
            message = f'{path}: the path walked through a non-dataclass member at {part!r}'
            raise ValueError(message)
        hints = get_type_hints(holder, include_extras=True)
        if part not in hints:
            message = f'{path}: {holder.__name__} has no field {part!r}'
            raise ValueError(message)
        annotation = hints[part]
        if position == len(parts) - 1:
            return annotation
        prefix = '.'.join(parts[: position + 1])
        choices = union_choices(annotation=annotation)
        if choices is not None:
            name: object = settings.get(prefix)
            if name is None:
                name = _default_choice_name(holder=holder, field_name=part)
            if not isinstance(name, str) or name not in choices:
                if name is None:
                    message = f'{path}: choose {part} first'
                else:
                    known = ', '.join(sorted(choices))
                    message = f'{path}: {name!r} is not a choice; choices are {known}'
                raise ValueError(message)
            holder = choices[name]
        else:
            member = _strip(annotation=annotation)
            if not (isinstance(member, type) and dataclasses.is_dataclass(member)):
                message = f'{path}: {part} is not a dataclass field to walk into'
                raise ValueError(message)
            holder = member
    return annotation


def settings_to_args(*, config_class: type, settings: Settings) -> list[str]:
    """Turn ``settings`` into the flags tyro reads back into ``config_class``.

    A path that ends at a choice union becomes the subcommand word
    ``<path>:<name>``; every other path becomes ``--<flag>`` plus its value.
    Words come first, parents before nested choices, then the flags in the
    order of ``settings``.

    Args:
        config_class: The config dataclass the settings are for.
        settings: The settings mapping to turn into flags.

    Returns:
        The flag list, ready to append to the command of a ``no-preset`` run.

    Raises:
        ValueError: If a path is unknown, a choice name is unknown, an
            unchosen union has no default member, or the text ``'None'`` is
            given for a field that accepts ``None``.
        TypeError: If a value is not a settings value.
    """
    words: list[str] = []
    flags: list[str] = []
    for path, value in settings.items():
        annotation = field_annotation(config_class=config_class, path=path, settings=settings)
        choices = union_choices(annotation=annotation)
        if choices is not None:
            if not isinstance(value, str) or value not in choices:
                known = ', '.join(sorted(choices))
                message = f'{path}: {value!r} is not a choice; choices are {known}'
                raise ValueError(message)
            words.append(f'{path}:{value}')
        elif _strip(annotation=annotation) is bool:
            if not isinstance(value, bool):
                message = f'{path}: expected a bool, got {type(value).__qualname__}'
                raise TypeError(message)
            flags += _bool_flag(path=path, value=value)
        else:
            if isinstance(value, str) and value == 'None' and _allows_none(annotation=annotation):
                message = (
                    f'{path}: the text None is ambiguous; pass the Python value '
                    f'None (not the string) to set no value'
                )
                raise ValueError(message)
            flags.append(_flag(path=path))
            flags += _value_args(value=value, path=path)
    words.sort(key=lambda word: word.count('.'))
    return words + flags

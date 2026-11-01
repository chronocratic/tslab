"""Typed dotted settings: the declarative twin of command-line flags.

``Settings`` maps a dotted field path of the config class of the script to a value
(``{'model': 'ts2vec', 'model.depth': 10}``). ``settings_to_args`` turns them
into the flags tyro reads back into the same config; the round trip
settings to flags to ``tslab.parse_config`` is pinned by tests.
"""

from collections.abc import Mapping, Sequence
import dataclasses
from dataclasses import Field, fields, MISSING
from enum import Enum
from pathlib import Path
import types
from typing import Any, cast, get_args, get_origin, get_type_hints, Union

from tslab.core.codec import strip_annotated, union_choices
from tslab.core.config import REQUIRED

__all__ = [
    'SettingValue',
    'Settings',
    'check_paths',
    'field_annotation',
    'from_presets',
    'settings_to_args',
]

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
    annotation = strip_annotated(annotation=annotation)
    if annotation is Any:
        return True
    if get_origin(annotation) in {Union, types.UnionType}:
        return any(member is type(None) for member in get_args(annotation))
    return False


def _default_choice_name(*, holder: type, field_name: str) -> str | None:
    """The subcommand name of the default instance of a union field, if it has one."""
    dc_fields = dataclasses.fields(cast('Any', holder))  # a dataclass, checked by the caller
    field = next(item for item in dc_fields if item.name == field_name)
    instance: object = None
    if field.default is not MISSING:
        instance = field.default
    elif field.default_factory is not MISSING:
        instance = field.default_factory()
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
    an unchosen union uses the default member of the union field when it has one.

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
    for position, part in enumerate(parts):
        owner = _dataclass_holder(path=path, holder=holder, part=part)
        annotation = _field_hint(path=path, holder=owner, part=part)
        if position == len(parts) - 1:
            return annotation
        holder = _next_holder(
            path=path,
            holder=owner,
            part=part,
            annotation=annotation,
            chosen=settings.get('.'.join(parts[: position + 1])),
        )
    return Any


def _dataclass_holder(*, path: str, holder: object, part: str) -> type:
    """The holder as a dataclass type, or a ``ValueError`` naming the part."""
    if not (isinstance(holder, type) and dataclasses.is_dataclass(holder)):
        message = f'{path}: the path walked through a non-dataclass member at {part!r}'
        raise ValueError(message)
    return holder


def _field_hint(*, path: str, holder: type, part: str) -> object:
    """The annotation of one field of a dataclass."""
    hints = get_type_hints(holder, include_extras=True)
    if part not in hints:
        message = f'{path}: {holder.__name__} has no field {part!r}'
        raise ValueError(message)
    return hints[part]


def _plain_member(*, path: str, part: str, annotation: object) -> type:
    """The dataclass of a field that is not a choice union, or a ``ValueError``."""
    member = strip_annotated(annotation=annotation)
    if not (isinstance(member, type) and dataclasses.is_dataclass(member)):
        message = f'{path}: {part} is not a dataclass field to walk into'
        raise ValueError(message)
    return member


def _next_holder(
    *, path: str, holder: type, part: str, annotation: object, chosen: object
) -> object:
    """The dataclass the path walks into below one field: the chosen member of a union."""
    choices = union_choices(annotation=annotation)
    if choices is None:
        return _plain_member(path=path, part=part, annotation=annotation)
    name = chosen if chosen is not None else _default_choice_name(holder=holder, field_name=part)
    if name is None:
        message = f'{path}: choose {part} first'
        raise ValueError(message)
    if not isinstance(name, str) or name not in choices:
        known = ', '.join(sorted(choices))
        message = f'{path}: {name!r} is not a choice; choices are {known}'
        raise ValueError(message)
    return choices[name]


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
            unchosen union has no default member, ``None`` is given for a
            field that does not accept it, or the text ``'None'`` is given for
            a field that does.
        TypeError: If a value is not a settings value.
    """
    words: list[str] = []
    flags: list[str] = []
    for path, value in settings.items():
        annotation = field_annotation(config_class=config_class, path=path, settings=settings)
        choices = union_choices(annotation=annotation)
        if choices is not None:
            words.append(_choice_word(path=path, value=value, choices=choices))
        elif strip_annotated(annotation=annotation) is bool:
            flags += _bool_flags(path=path, value=value)
        else:
            flags += _leaf_flags(path=path, value=value, annotation=annotation)
    words.sort(key=lambda word: word.count('.'))
    return words + flags


def _choice_word(*, path: str, value: SettingValue, choices: Mapping[str, type]) -> str:
    """The subcommand word of a choice union path."""
    if not isinstance(value, str) or value not in choices:
        known = ', '.join(sorted(choices))
        message = f'{path}: {value!r} is not a choice; choices are {known}'
        raise ValueError(message)
    return f'{path}:{value}'


def _bool_flags(*, path: str, value: SettingValue) -> list[str]:
    """The flag words of a bool path."""
    if not isinstance(value, bool):
        message = f'{path}: expected a bool, got {type(value).__qualname__}'
        raise TypeError(message)
    return _bool_flag(path=path, value=value)


def _leaf_flags(*, path: str, value: SettingValue, annotation: object) -> list[str]:
    """The flag words of a leaf path that is neither a choice nor a bool."""
    if value is None and not _allows_none(annotation=annotation):
        message = f'{path}: None is not a value of a field that does not accept None'
        raise ValueError(message)
    if isinstance(value, str) and value == 'None' and _allows_none(annotation=annotation):
        message = (
            f'{path}: the text None is ambiguous; pass the Python value '
            f'None (not the string) to set no value'
        )
        raise ValueError(message)
    if isinstance(value, str) and value.startswith('-'):
        # why: a separate word would be read as another flag
        return [f'{_flag(path=path)}={value}']
    return [_flag(path=path), *_value_args(value=value, path=path)]


def from_presets(
    *,
    presets: Mapping[str, tuple[str, object]],
    names: Sequence[str] | None = None,
    without: Sequence[str],
) -> dict[str, dict[str, SettingValue]]:
    """Turn presets of the script into settings, one mapping per preset.

    A preset gives, for every choice union field, the name of the chosen
    member, and for every other field, the value only where it differs from
    the default tyro starts from (the default instance of the field when it
    has one, else the default of the class), plus every field that has no
    default. Fields holding ``tslab.REQUIRED`` are skipped: the study must
    supply them on an axis. Every path under a path named in ``without`` is
    dropped too.

    Args:
        presets: The preset registry of the script, ``{name: (description, config)}``.
        names: Which presets to read, in this order; ``None`` reads all of
            them in registry order.
        without: Field paths to drop, e.g. ``('data',)``: the path and every
            path below it. Required, with no default.

    Returns:
        ``{preset name: settings}`` in the order of ``names``.

    Raises:
        ValueError: If a name in ``names`` is not in ``presets``, or a path in
            ``without`` is no field of any selected preset.
        TypeError: If a preset holds a leaf value settings cannot hold, or a
            choice union field holds a non-member.
    """
    selected = _selected_presets(presets=presets, names=names)
    unknown_without = [
        drop
        for drop in without
        if not any(_has_path(config=presets[name][1], path=drop) for name in selected)
    ]
    if unknown_without:
        message = (
            f'without names {unknown_without}, which is no field of any selected preset; '
            f'check the spelling'
        )
        raise ValueError(message)
    result: dict[str, dict[str, SettingValue]] = {}
    for name in selected:
        _, config = presets[name]
        settings = _preset_settings(config=config, base=None, prefix='', preset_name=name)
        result[name] = {
            path: value
            for path, value in settings.items()
            if not any(path == drop or path.startswith(f'{drop}.') for drop in without)
        }
    return result


def _selected_presets(
    *, presets: Mapping[str, tuple[str, object]], names: Sequence[str] | None
) -> list[str]:
    """The names of the presets to read, in order; ``ValueError`` for an unknown one."""
    if names is None:
        return list(presets)
    unknown = [name for name in names if name not in presets]
    if unknown:
        message = f'unknown preset names {unknown}; known: {sorted(presets)}'
        raise ValueError(message)
    return list(names)


_NO_DEFAULT = object()


def _field_default(*, field: Field[Any]) -> object:
    if field.default is not MISSING:
        return field.default
    if field.default_factory is not MISSING:
        return field.default_factory()
    return _NO_DEFAULT


def _is_setting_value(*, value: object) -> bool:
    if value is None or isinstance(value, str | int | float | bool | Path | Enum):
        return True
    if isinstance(value, tuple | list):
        return all(_is_setting_value(value=item) for item in value)
    return False


def _preset_leaf(
    *, item: object, path: str, preset_name: str, default: object
) -> dict[str, SettingValue]:
    if item is REQUIRED:
        return {}
    if default is not _NO_DEFAULT and item == default:
        return {}
    if not _is_setting_value(value=item):
        message = (
            f'preset {preset_name}: {path} holds a value of type '
            f'{type(item).__qualname__} that settings cannot hold'
        )
        raise TypeError(message)
    return {path: cast('SettingValue', item)}


def _has_path(*, config: object, path: str) -> bool:
    """Whether the dotted field path exists on the preset config."""
    current = config
    for part in path.split('.'):
        if not dataclasses.is_dataclass(current) or isinstance(current, type):
            return False
        if part not in {item.name for item in fields(current)}:
            return False
        current = getattr(current, part)
    return True


def _default_of(*, field: Field[Any], base: object | None) -> object:
    """The default tyro starts from: the field of the base instance, else the class default."""
    if base is not None:
        return getattr(base, field.name)
    return _field_default(field=field)


def _preset_settings(
    *, config: object, base: object | None, prefix: str, preset_name: str
) -> dict[str, SettingValue]:
    if not dataclasses.is_dataclass(config) or isinstance(config, type):
        message = f'preset {preset_name} holds a non-dataclass at {prefix or "the top level"}'
        raise TypeError(message)
    cls = type(config)
    hints = get_type_hints(cls, include_extras=True)
    settings: dict[str, SettingValue] = {}
    for field in fields(cls):
        if not field.init:
            continue
        settings.update(
            _field_settings(
                item=getattr(config, field.name),
                default=_default_of(field=field, base=base),
                annotation=hints[field.name],
                path=f'{prefix}.{field.name}' if prefix else field.name,
                preset_name=preset_name,
            )
        )
    return settings


def _field_settings(
    *, item: object, default: object, annotation: object, path: str, preset_name: str
) -> dict[str, SettingValue]:
    """The settings of one field of a preset: a choice, a nested dataclass or a leaf."""
    choices = union_choices(annotation=annotation)
    if choices is not None:
        return _choice_settings(
            item=item, default=default, choices=choices, path=path, preset_name=preset_name
        )
    if dataclasses.is_dataclass(item) and not isinstance(item, type):
        return _nested_settings(item=item, default=default, path=path, preset_name=preset_name)
    return _preset_leaf(item=item, path=path, preset_name=preset_name, default=default)


def _choice_settings(
    *, item: object, default: object, choices: Mapping[str, type], path: str, preset_name: str
) -> dict[str, SettingValue]:
    """The settings of a choice union field: the chosen name, then the member settings."""
    choice_name = next(
        (member_name for member_name, member_cls in choices.items() if type(item) is member_cls),
        None,
    )
    if choice_name is None:
        message = (
            f'preset {preset_name}: {path} holds {type(item).__qualname__}, '
            f'which is not a member of the choice'
        )
        raise TypeError(message)
    settings: dict[str, SettingValue] = {path: choice_name}
    if item is not None:
        # why: tyro starts a chosen member from the default instance only when the
        # default is a member of the same class; otherwise from the class defaults
        member_base = default if type(default) is type(item) else None
        settings.update(
            _preset_settings(config=item, base=member_base, prefix=path, preset_name=preset_name)
        )
    return settings


def _nested_settings(
    *, item: object, default: object, path: str, preset_name: str
) -> dict[str, SettingValue]:
    """The settings of a nested dataclass field; none when it equals the default."""
    if default is not _NO_DEFAULT and item == default:
        return {}
    nested_base = default if type(default) is type(item) else None
    return _preset_settings(config=item, base=nested_base, prefix=path, preset_name=preset_name)


def check_paths(*, config_class: type, settings: Settings) -> list[str]:
    """The paths of ``settings`` that exist on no reading of ``config_class``.

    Walks like ``field_annotation`` without raising: at a choice union that
    is not chosen in the same mapping, the path is accepted when it exists
    in any member (the choice may come from another axis); a chosen union
    walks only the chosen member. A choice value that is not a member name
    is a bad path, so the path of the union field itself is reported.

    Args:
        config_class: The config dataclass the paths are relative to.
        settings: The settings mapping whose paths are checked.

    Returns:
        The bad paths in the order of ``settings``; empty when all are fine.
    """
    return [
        path
        for path in settings
        if not _path_exists(path=path, config_class=config_class, settings=settings)
    ]


def _path_exists(*, path: str, config_class: type, settings: Settings) -> bool:
    parts = path.split('.')
    holders: list[type] = [config_class]
    for position, part in enumerate(parts):
        name = settings.get('.'.join(parts[: position + 1]))
        if position == len(parts) - 1:
            return any(_final_exists(holder=holder, part=part, name=name) for holder in holders)
        next_holders: list[type] = []
        for holder in holders:
            next_holders.extend(_steps_into(holder=holder, part=part, name=name))
        holders = next_holders
        if not holders:
            return False
    return True


def _steps_into(*, holder: type, part: str, name: object) -> list[type]:
    """The dataclass holders below a field, all members of an unchosen union."""
    hints = get_type_hints(holder, include_extras=True)
    if part not in hints:
        return []
    annotation = hints[part]
    choices = union_choices(annotation=annotation)
    if choices is None:
        member = strip_annotated(annotation=annotation)
        return [member] if isinstance(member, type) and dataclasses.is_dataclass(member) else []
    if isinstance(name, str) and name in choices:
        return [choices[name]]
    return list(choices.values())


def _final_exists(*, holder: type, part: str, name: object) -> bool:
    hints = get_type_hints(holder, include_extras=True)
    if part not in hints:
        return False
    choices = union_choices(annotation=hints[part])
    if choices is None:
        return True
    return not (isinstance(name, str) and name not in choices)

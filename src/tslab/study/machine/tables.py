"""The shared TOML table checks of a machine profile."""

import dataclasses
import os
from pathlib import Path
import re
from typing import cast

from tslab.core.config import ConfigFileError, decode_dataclass
from tslab.core.text import unknown_key_message
from tslab.study.expansion import expand_variables
from tslab.study.settings import SettingValue

__all__ = [
    'MachineProfileError',
    'check_keys',
    'check_range',
    'decode',
    'expand_env',
    'flatten_settings',
    'from_table',
    'is_setting_value',
    'settings_of',
    'string_field',
    'string_tuple',
]

_ENV_NAME_PATTERN = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')

# why: the smallest value each numeric key accepts; the keys are unique across the tables
_MINIMUMS: dict[str, float] = {
    'cpus_per_task': 1,
    'gpus': 0,
    'signal_seconds': 0,
    'max_concurrent': 1,
    'max_array_size': 1,
    'submit_pause_seconds': 0,
    'max_attempts': 1,
    'max_parallel': 1,
    'tasks_per_gpu': 1,
}


class MachineProfileError(ValueError):
    """The profile file does not hold a machine profile."""


def from_table[T](
    *, cls: type[T], table: object, location: str, path: Path, required: bool
) -> T | None:
    """Decode ``table`` into an instance of ``cls``, checking keys, types and ranges.

    Args:
        cls: The data class to decode the table into.
        table: The raw TOML table, or ``None`` for an absent optional table.
        location: The dotted name of the table, for errors.
        path: The profile file, named in every error.
        required: Whether ``table`` must be present.

    Returns:
        The decoded instance, or ``None`` when an optional table is absent.
    """
    if table is None:
        if required:
            message = f'{path}: the {location} table is required'
            raise MachineProfileError(message)
        return None
    if not isinstance(table, dict):
        message = f'{path}: {location} must be a table, got {type(table).__name__}'
        raise MachineProfileError(message)
    check_keys(cls=cls, table=table, location=location, path=path)
    instance = decode(data=table, cls=cls, location=location, path=path)
    for key in table:
        check_range(key=key, value=getattr(instance, key), location=f'{location}.{key}', path=path)
    return instance


def decode[T](*, data: dict[str, object], cls: type[T], location: str, path: Path) -> T:
    """Decode ``data`` into ``cls``; a config error becomes a profile error that names the file."""
    try:
        return decode_dataclass(data=data, cls=cls, location=location)
    except ConfigFileError as error:
        message = f'{path}: {error}'
        raise MachineProfileError(message) from error


def check_keys(*, cls: type, table: dict[str, object], location: str, path: Path) -> None:
    """Raise if ``table`` holds a key that ``cls`` does not have.

    Args:
        cls: The data class whose fields are the valid keys.
        table: The raw TOML table to check.
        location: The dotted name of the table, for the error.
        path: The profile file, named in the error.
    """
    known = {item.name for item in dataclasses.fields(cls)}  # ty: ignore[invalid-argument-type]
    unknown = [name for name in table if name not in known]
    if unknown:
        problem = unknown_key_message(location=location, key=unknown[0], valid=known)
        message = f'{path}: {problem}'
        raise MachineProfileError(message)


def check_range(*, key: str, value: object, location: str, path: Path) -> None:
    """Numbers stay in range; text that goes into a job script holds no line break."""
    minimum = _MINIMUMS.get(key)
    if minimum is not None and isinstance(value, int | float) and value < minimum:
        message = f'{path}: {location} must be at least {minimum}, got {value}'
        raise MachineProfileError(message)
    texts = [value] if isinstance(value, str) else list(value) if isinstance(value, tuple) else []
    if any(isinstance(text, str) and re.search(r'[\x00-\x1f\x7f]', text) for text in texts):
        message = f'{path}: {location} must not hold a line break or other control character'
        raise MachineProfileError(message)


def string_field(*, table: dict[str, object], name: str, path: Path) -> str:
    """The required string field ``name`` of ``table``.

    Args:
        table: The raw TOML table to read.
        name: The field to read.
        path: The profile file, named in the error.
    """
    if name not in table:
        message = f'{path}: {name} is required'
        raise MachineProfileError(message)
    value = table[name]
    if not isinstance(value, str):
        message = f'{path}: {name} must be a string, got {type(value).__name__}'
        raise MachineProfileError(message)
    return value


def string_tuple(
    *, table: dict[str, object], name: str, path: Path, default: tuple[str, ...]
) -> tuple[str, ...]:
    """The field ``name`` of ``table``: a list of strings, returned as a tuple.

    Args:
        table: The raw TOML table to read.
        name: The field to read.
        path: The profile file, named in the error.
        default: The value when the field is absent.
    """
    value = table.get(name, list(default))
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        message = f'{path}: {name} must be a list of strings'
        raise MachineProfileError(message)
    return tuple(cast('list[str]', value))


def expand_env(*, table: dict[str, object], path: Path, lenient: bool) -> dict[str, str]:
    """Expand the values of the ``[env]`` table of ``table`` from the environment.

    Args:
        table: The raw TOML table that holds the ``[env]`` table.
        path: The profile file, named in the error.
        lenient: Keep a value with an unset variable as it is instead of raising.

    Returns:
        The expanded name to value mapping.
    """
    value = table.get('env', {})
    if not isinstance(value, dict) or any(not isinstance(item, str) for item in value.values()):
        message = f'{path}: [env] must hold string values'
        raise MachineProfileError(message)
    expanded: dict[str, str] = {}
    for name, item in value.items():
        if _ENV_NAME_PATTERN.fullmatch(name) is None:
            message = (
                f'{path}: [env] name {name!r} is not a shell variable name '
                f'(letters, digits and underscores, not starting with a digit)'
            )
            raise MachineProfileError(message)
        try:
            expanded[name] = expand_variables(text=item, env={**os.environ, **expanded})
        except ValueError as error:
            if not lenient:
                message = f'{path}: [env] value for {name!r}: {error}'
                raise MachineProfileError(message) from error
            expanded[name] = item
    return expanded


def settings_of(*, table: dict[str, object], location: str, path: Path) -> dict[str, SettingValue]:
    """The flattened ``settings`` table of ``table``; every value must be a settings value."""
    value = table.get('settings', {})
    if not isinstance(value, dict):
        message = f'{path}: {location} must be a table, got {type(value).__name__}'
        raise MachineProfileError(message)
    flat = flatten_settings(table=value)
    for key, item in flat.items():
        if not is_setting_value(value=item):
            message = (
                f'{path}: {location}.{key} holds {type(item).__name__}, which settings cannot '
                f'hold; use strings, numbers, booleans or lists of those'
            )
            raise MachineProfileError(message)
    return flat


def is_setting_value(*, value: object) -> bool:
    """Whether ``value`` is a settings value: a string, number or boolean, or a list of those."""
    if isinstance(value, str | int | float | bool):
        return True
    return isinstance(value, tuple) and all(is_setting_value(value=item) for item in value)


def flatten_settings(*, table: dict[str, object], prefix: str = '') -> dict[str, SettingValue]:
    """A settings table as dotted paths; lists become tuples.

    A nested table ``[settings.tracking]`` with ``wandb_mode = "offline"`` is
    the same as ``"tracking.wandb_mode" = "offline"``.

    Args:
        table: The table to flatten.
        prefix: The dotted prefix of this table, for recursion.

    Returns:
        ``{dotted path: value}`` with every list a tuple.
    """
    flat: dict[str, SettingValue] = {}
    for key, value in table.items():
        dotted = f'{prefix}.{key}' if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_settings(table=value, prefix=dotted))
        else:
            flat[dotted] = tuple(value) if isinstance(value, list) else cast('SettingValue', value)
    return flat

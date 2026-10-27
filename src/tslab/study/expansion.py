"""String expansion for study settings: ``${VAR}`` and ``{placeholder}``."""

from collections.abc import Callable, Mapping
from enum import Enum
from pathlib import Path
import re
from typing import cast

from tslab.study.settings import SettingValue

__all__ = ['PLACEHOLDERS', 'expand_placeholders', 'expand_setting', 'expand_variables']

PLACEHOLDERS = ('output_root', 'study', 'run_path', 'log_file', 'cpus')

_VARIABLE_PATTERN = re.compile(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}')
_PLACEHOLDER_PATTERN = re.compile(r'\{\{|\}\}|\{([A-Za-z_][A-Za-z0-9_]*)\}')
_BOTH_PATTERN = re.compile(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\{\{|\}\}|\{([A-Za-z_][A-Za-z0-9_]*)\}')


def expand_variables(*, text: str, env: Mapping[str, str]) -> str:
    """Every ``${NAME}`` in ``text`` replaced with ``env[NAME]``.

    A ``$`` not followed by ``{NAME}`` is kept as it is.

    Args:
        text: The text to expand.
        env: The variable names to values; the profile ``[env]`` on top of
            the process environment.

    Returns:
        The text with every ``${NAME}`` replaced.

    Raises:
        ValueError: If a named variable is not in ``env``.
    """
    return _VARIABLE_PATTERN.sub(_replace_variable(env=env), text)


def _replace_variable(*, env: Mapping[str, str]) -> Callable[[re.Match[str]], str]:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in env:
            message = f'environment variable ${{{name}}} is not set'
            raise ValueError(message)
        return env[name]

    return replace


def expand_placeholders(*, text: str, values: Mapping[str, str]) -> str:
    """Every ``{name}`` in ``text`` replaced with ``values[name]``.

    Only the names in ``PLACEHOLDERS`` are accepted; any other ``{...}``
    raises, so a typo in a study setting cannot hide in a path. ``{{`` and
    ``}}`` stand for a literal brace, e.g. for a checkpoint filename template.

    Args:
        text: The text to expand.
        values: The placeholder names to their values for this run.

    Returns:
        The text with every placeholder replaced.

    Raises:
        ValueError: If the text holds a ``{name}`` that is not one of
            ``PLACEHOLDERS`` or has no value.
    """
    return _PLACEHOLDER_PATTERN.sub(_replace_placeholder(values=values), text)


def _replace_placeholder(*, values: Mapping[str, str]) -> Callable[[re.Match[str]], str]:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name is None:
            return match.group(0)[0]
        if name not in PLACEHOLDERS or name not in values:
            message = f'{name!r} is not a placeholder value for this run'
            raise ValueError(message)
        return values[name]

    return replace


def _expand_text(*, text: str, env: Mapping[str, str], values: Mapping[str, str]) -> str:
    """Variables and placeholders in one pass, so a value is never scanned again."""
    variable = _replace_variable(env=env)
    placeholder = _replace_placeholder(values=values)

    def replace(match: re.Match[str]) -> str:
        if match.group(1) is not None:
            return variable(match)
        return placeholder(cast('re.Match[str]', _PLACEHOLDER_PATTERN.fullmatch(match.group(0))))

    return _BOTH_PATTERN.sub(replace, text)


def expand_setting(
    *, value: SettingValue, env: Mapping[str, str], values: Mapping[str, str]
) -> SettingValue:
    """A setting value with its strings, paths and sequence items expanded.

    Variables and placeholders are expanded in one pass on ``str`` values, on the
    POSIX text of ``Path`` values (the result is a ``Path`` again) and on the
    string items of tuples and lists; every other value is returned unchanged.

    Args:
        value: The setting value to expand.
        env: The variable names to values.
        values: The placeholder names to their values for this run.

    Returns:
        The value with every expandable string expanded.

    Raises:
        ValueError: If a string holds an unset variable or an unknown
            placeholder.
    """
    if isinstance(value, Enum):  # a str Enum is a str, but expanding it would lose the member
        return value
    if isinstance(value, str):
        return _expand_text(text=value, env=env, values=values)
    if isinstance(value, Path):
        return Path(_expand_text(text=value.as_posix(), env=env, values=values))
    if isinstance(value, tuple):
        expanded = tuple(_expand_item(item=item, env=env, values=values) for item in value)
        return cast('SettingValue', expanded)
    if isinstance(value, list):
        expanded = [_expand_item(item=item, env=env, values=values) for item in value]
        return cast('SettingValue', expanded)
    return value


def _expand_item(*, item: object, env: Mapping[str, str], values: Mapping[str, str]) -> object:
    if isinstance(item, str):
        return _expand_text(text=item, env=env, values=values)
    if isinstance(item, Path):
        return Path(_expand_text(text=item.as_posix(), env=env, values=values))
    return item

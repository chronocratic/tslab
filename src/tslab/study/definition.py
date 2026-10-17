"""The study definition: a round of experiments as grids of typed settings.

A ``Study`` is named grids; a ``Grid`` is one script command, its config
class and named axes of settings, expanded into the runs of the study.
Validation happens in the constructors, so a study module that fails to
import fails at once, with the problem named.
"""

from collections.abc import Mapping, Sequence
import dataclasses
from dataclasses import dataclass
import re
from types import MappingProxyType

from tslab.study.settings import Settings, SettingValue

__all__ = ['DEFAULT_RUN_SETTINGS', 'Grid', 'Match', 'Override', 'Study', 'setting_axis']

type Match = Mapping[str, str | Sequence[str]]

_NAME_PATTERN = re.compile(r'[A-Za-z0-9._-]+')
_SHORT_NAME_PATTERN = re.compile(r'[A-Za-z0-9_-]{1,8}')

DEFAULT_RUN_SETTINGS: Mapping[str, str] = MappingProxyType(
    {'output_root': '{output_root}/{study}', 'run_path': '{run_path}'}
)


def validate_name(*, name: str, kind: str) -> str:
    """Check a study, grid, axis or value name; return it when valid.

    Args:
        name: The name to check.
        kind: What the name is, for the error message, e.g. ``'grid'``.

    Returns:
        The name, unchanged.

    Raises:
        ValueError: If the name is empty, ``'.'`` or ``'..'``, or holds any
            character outside ``[A-Za-z0-9._-]``.
    """
    if not name or name in {'.', '..'} or _NAME_PATTERN.fullmatch(name) is None:
        message = (
            f'invalid {kind} name {name!r}; names use letters, digits, dot, dash or underscore'
        )
        raise ValueError(message)
    return name


def setting_axis(
    *, path: str, values: Sequence[SettingValue]
) -> dict[str, dict[str, SettingValue]]:
    """An axis that varies one setting over the given values.

    Args:
        path: The dotted field path to vary.
        values: The values of the setting, in the order of the axis.

    Returns:
        ``{str(value): {path: value}}`` in the order of ``values``.
    """
    return {str(value): {path: value} for value in values}


@dataclass(frozen=True, kw_only=True)
class Override:
    """Settings applied to the runs whose axis values match."""

    match: Match
    settings: Settings


@dataclass(frozen=True, kw_only=True)
class Grid:
    """One script command of a study, run over the product of its axes.

    Args:
        short_name: At most 8 characters, unique in the study; shown where a
            full grid name does not fit, e.g. on ``squeue``.
        command: The no-preset command of the script, as a word tuple.
        config: The script's config class the settings are typed against.
        axes: Axis name to ``{value name: settings}``, in run-path order.
        exclude: Matches naming the runs to drop.
        overrides: Settings applied to matching runs.
        run_settings: Settings every run of the grid gets, e.g. the run
            paths. Read-only once built.
    """

    short_name: str
    command: tuple[str, ...]
    config: type
    axes: Mapping[str, Mapping[str, Settings]]
    exclude: tuple[Match, ...] = ()
    overrides: tuple[Override, ...] = ()
    run_settings: Settings = DEFAULT_RUN_SETTINGS

    def __post_init__(self) -> None:
        """Check the shape: names, command, config, axes."""
        if _SHORT_NAME_PATTERN.fullmatch(self.short_name) is None:
            message = (
                f'invalid short_name {self.short_name!r}; '
                f'use at most 8 letters, digits or underscores (squeue shows 8)'
            )
            raise ValueError(message)
        if not self.command:
            message = 'grid command must not be empty'
            raise ValueError(message)
        if not (isinstance(self.config, type) and dataclasses.is_dataclass(self.config)):
            message = f'grid config must be a dataclass, got {type(self.config).__qualname__}'
            raise ValueError(message)
        if not self.axes:
            message = 'grid must have at least one axis'
            raise ValueError(message)
        for axis_name, values in self.axes.items():
            validate_name(name=axis_name, kind='axis')
            if not values:
                message = f'axis {axis_name!r} must have at least one value'
                raise ValueError(message)
            for value_name in values:
                validate_name(name=value_name, kind=f'axis {axis_name!r} value')


@dataclass(frozen=True, kw_only=True)
class Study:
    """A named round of experiments: a mapping of grid name to grid."""

    name: str
    grids: Mapping[str, Grid]

    def __post_init__(self) -> None:
        """Check the study name, the grid names and the short names."""
        validate_name(name=self.name, kind='study')
        if not self.grids:
            message = 'study must have at least one grid'
            raise ValueError(message)
        short_names: dict[str, str] = {}
        for grid_name, grid in self.grids.items():
            validate_name(name=grid_name, kind='grid')
            if grid.short_name in short_names:
                message = (
                    f'grid {grid_name!r} reuses short_name {grid.short_name!r} '
                    f'of grid {short_names[grid.short_name]!r}'
                )
                raise ValueError(message)
            short_names[grid.short_name] = grid_name

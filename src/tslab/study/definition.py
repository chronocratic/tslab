"""The study definition: a round of experiments as grids of typed settings.

A ``Study`` is named grids; a ``Grid`` is one script command, its config
class and named axes of settings, expanded into the runs of the study.
Validation happens in the constructors, so a study module that fails to
import fails at once, with the problem named.
"""

from collections.abc import Mapping, Sequence
import dataclasses
from dataclasses import dataclass
from itertools import product
import re
from types import MappingProxyType

from tslab.study.settings import check_paths, Settings, SettingValue

__all__ = [
    'DEFAULT_RUN_SETTINGS',
    'Grid',
    'Match',
    'Override',
    'Study',
    'match_problems',
    'normalize_match',
    'run_matches',
    'select_runs',
    'setting_axis',
]

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
            problems = [
                problem
                for match in (*grid.exclude, *(override.match for override in grid.overrides))
                for problem in match_problems(grids={grid_name: grid}, match=match)
            ]
            if problems:
                message = f'grid {grid_name!r}: ' + '; '.join(problems)
                raise ValueError(message)
            if grid.short_name in short_names:
                message = (
                    f'grid {grid_name!r} reuses short_name {grid.short_name!r} '
                    f'of grid {short_names[grid.short_name]!r}'
                )
                raise ValueError(message)
            short_names[grid.short_name] = grid_name
            _grid_settings_problems(grid_name=grid_name, grid=grid)
        if not any(_grid_has_runs(grid_name=name, grid=grid) for name, grid in self.grids.items()):
            message = 'study has no runs; every run is excluded'
            raise ValueError(message)


def normalize_match(*, match: Match) -> dict[str, tuple[str, ...]]:
    """A match with every value as a tuple of value names.

    Args:
        match: Axis name to value name, or list of value names; a ``grid``
            key names a grid.

    Returns:
        The same match with every value a tuple.

    Raises:
        ValueError: If a value list is empty.
        TypeError: If a value list holds a non-string.
    """
    normalized: dict[str, tuple[str, ...]] = {}
    for key, value in match.items():
        values = (value,) if isinstance(value, str) else tuple(value)
        if not values:
            message = f'match key {key!r} has no values'
            raise ValueError(message)
        for item in values:
            if not isinstance(item, str):
                message = f'match key {key!r} holds a non-string value {item!r}'
                raise TypeError(message)
        normalized[key] = values
    return normalized


def match_problems(*, grids: Mapping[str, Grid], match: Match) -> list[str]:
    """The human-readable problems a match has against the given grids.

    A key other than ``grid`` must be an axis of at least one grid, and its
    value names must be value names of that axis in at least one grid; a
    ``grid`` value must name one of the grids.

    Args:
        grids: The grids the match is checked against.
        match: The match to check.

    Returns:
        The problems, one line each; empty when the match is fine.
    """
    problems: list[str] = []
    for key, value in normalize_match(match=match).items():
        if key == 'grid':
            unknown = [name for name in value if name not in grids]
            if unknown:
                problems.append(f'grid {", ".join(unknown)!r} is not a grid of this study')
            continue
        axis_grids = [grid for grid in grids.values() if key in grid.axes]
        if not axis_grids:
            problems.append(f'{key} is not an axis of any grid')
            continue
        known = {name for grid in axis_grids for name in grid.axes[key]}
        unknown = [name for name in value if name not in known]
        if unknown:
            problems.append(f'{key} has no value {", ".join(unknown)!r}')
    return problems


def run_matches(*, grid_name: str, values: Mapping[str, str], match: Match) -> bool:
    """Whether a run of one grid matches: every key must hold.

    A key the run's grid has no axis for makes the run not match.

    Args:
        grid_name: The grid the run belongs to.
        values: The run's axis name to value name.
        match: The match to test.

    Returns:
        True when every key of the match holds for the run.
    """
    normalized = normalize_match(match=match)
    if grid_name not in normalized.get('grid', (grid_name,)):
        return False
    for key, value_names in normalized.items():
        if key != 'grid' and values.get(key) not in value_names:
            return False
    return True


@dataclass(frozen=True, kw_only=True)
class StudyRun:
    """One run of a grid: its axis values, run path and merged settings."""

    grid: str
    values: Mapping[str, str]
    run_path: str
    settings: Mapping[str, SettingValue]


def merge_overrides(
    *,
    base: Mapping[str, SettingValue],
    overrides: Sequence[tuple[Match, Settings]],
    grid_name: str,
    values: Mapping[str, str],
) -> dict[str, SettingValue]:
    """Apply every matching override to ``base``, in precedence order.

    Overrides are applied by number of match keys, then position: an override
    with more match keys wins over one with fewer, and between equal ones the
    later is applied last.

    Args:
        base: The settings to apply the overrides to.
        overrides: Pairs of match and settings.
        grid_name: The grid the run belongs to.
        values: The run's axis name to value name.

    Returns:
        The base settings with the matching overrides applied in order.
    """
    result = dict(base)
    matching = [
        position
        for position, (match, _) in enumerate(overrides)
        if run_matches(grid_name=grid_name, values=values, match=match)
    ]
    matching.sort(key=lambda position: len(overrides[position][0]))
    for position in matching:
        result.update(overrides[position][1])
    return result


def expand_runs(*, study: Study) -> list[StudyRun]:
    """Expand a study into its runs, grids in order and axes in order.

    Per grid the runs are the product of its axes; excluded runs are dropped.

    Args:
        study: The study to expand.

    Returns:
        The runs, one per point of every grid's axis product that is not
        excluded, in grid order and then axis order.
    """
    runs: list[StudyRun] = []
    for grid_name, grid in study.grids.items():
        axis_names = list(grid.axes)
        for combo in product(*(grid.axes[name] for name in axis_names)):
            values = dict(zip(axis_names, combo, strict=True))
            if any(
                run_matches(grid_name=grid_name, values=values, match=match)
                for match in grid.exclude
            ):
                continue
            settings: dict[str, SettingValue] = {}
            for name in axis_names:
                settings.update(grid.axes[name][values[name]])
            settings.update(grid.run_settings)
            settings = merge_overrides(
                base=settings,
                overrides=[(override.match, override.settings) for override in grid.overrides],
                grid_name=grid_name,
                values=values,
            )
            runs.append(
                StudyRun(
                    grid=grid_name,
                    values=values,
                    run_path='/'.join([grid_name, *values.values()]),
                    settings=settings,
                )
            )
    return runs


def select_runs(
    *, study: Study, runs: Sequence[StudyRun], grids: Sequence[str] = (), only: Sequence[str] = ()
) -> list[StudyRun]:
    """The runs that pass the grid names and axis filters.

    An empty ``grids`` keeps every grid; a name that is not a grid raises.
    Each ``only`` item is ``AXIS=V1,V2``: it keeps the runs of the grids that
    have ``AXIS`` whose value for it is named; a malformed item, an axis no
    grid has or a value no grid's axis has raises. Filtering never changes
    run paths.

    Args:
        study: The study the runs came from.
        runs: The runs to filter, e.g. ``expand_runs``' output.
        grids: The grid names to keep; empty keeps them all.
        only: ``AXIS=V1,V2`` items, every one of which must hold.

    Returns:
        The runs that pass every filter, in the order of ``runs``.

    Raises:
        ValueError: If a grid name is unknown, an ``only`` item is malformed,
            an axis belongs to no grid or a value belongs to no grid's axis.
    """
    unknown_grids = [name for name in grids if name not in study.grids]
    if unknown_grids:
        message = (
            f'unknown grid {", ".join(unknown_grids)!r}; '
            f'this study has {", ".join(repr(name) for name in study.grids)}'
        )
        raise ValueError(message)
    parsed = [_parse_only(item=item, study=study) for item in only]
    return [
        run
        for run in runs
        if (not grids or run.grid in grids)
        and all(
            axis not in run.values or run.values[axis] in value_names
            for axis, value_names in parsed
        )
    ]


def _parse_only(*, item: str, study: Study) -> tuple[str, tuple[str, ...]]:
    """One ``AXIS=V1,V2`` item, checked against the study's grids."""
    axis, separator, values = item.partition('=')
    if not separator or not axis or not values:
        message = (
            f'only item {item!r} is malformed; items are an axis name as AXIS=V1,V2, '
            'where every V names a value of that axis'
        )
        raise ValueError(message)
    value_names = tuple(values.split(','))
    axis_grids = [grid for grid in study.grids.values() if axis in grid.axes]
    if not axis_grids:
        message = f'axis {axis!r} is not an axis of any grid of this study'
        raise ValueError(message)
    known = {name for grid in axis_grids for name in grid.axes[axis]}
    unknown = [name for name in value_names if name not in known]
    if unknown:
        message = f'axis {axis!r} has no value {", ".join(unknown)!r} in any grid'
        raise ValueError(message)
    return axis, value_names


def _grid_settings_problems(*, grid_name: str, grid: Grid) -> None:
    """Name the grid, the axis and value, or run_settings or override N, for each bad path."""
    for axis_name, values in grid.axes.items():
        for value_name, settings in values.items():
            bad = check_paths(config_class=grid.config, settings=settings)
            if bad:
                message = (
                    f'grid {grid_name!r}: axis {axis_name!r} value {value_name!r} '
                    f'sets paths that are not fields of {grid.config.__name__}: {", ".join(bad)}'
                )
                raise ValueError(message)
    bad = check_paths(config_class=grid.config, settings=grid.run_settings)
    if bad:
        message = (
            f'grid {grid_name!r}: run_settings set paths that are not fields '
            f'of {grid.config.__name__}: {", ".join(bad)}'
        )
        raise ValueError(message)
    for position, override in enumerate(grid.overrides):
        bad = check_paths(config_class=grid.config, settings=override.settings)
        if bad:
            message = (
                f'grid {grid_name!r}: override {position} sets paths that are not fields '
                f'of {grid.config.__name__}: {", ".join(bad)}'
            )
            raise ValueError(message)


def _grid_has_runs(*, grid_name: str, grid: Grid) -> bool:
    for combo in product(*(grid.axes[name] for name in grid.axes)):
        values = dict(zip(grid.axes, combo, strict=True))
        excluded = any(
            run_matches(grid_name=grid_name, values=values, match=match) for match in grid.exclude
        )
        if not excluded:
            return True
    return False

"""The study definition: a round of experiments as grids of typed settings.

A ``Study`` is named grids; a ``Grid`` is one script command, its config
class and named axes of settings, expanded into the runs of the study.
Validation happens in the constructors, so a study module that fails to
import fails at once, with the problem named.
"""

from collections.abc import Iterator, Mapping, Sequence
import dataclasses
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import re
from types import MappingProxyType
from typing import TYPE_CHECKING

from tslab.core.paths import NAME_PART_PATTERN
from tslab.study.settings import check_paths, Settings, SettingValue

if TYPE_CHECKING:
    from tslab.study.result_table import ResultRow

__all__ = [
    'DEFAULT_RUN_SETTINGS',
    'RESERVED_AXIS_NAMES',
    'Grid',
    'Match',
    'Override',
    'Study',
    'match_problems',
    'merge_settings',
    'normalize_match',
    'run_matches',
    'select_runs',
    'setting_axis',
]

type Match = Mapping[str, str | Sequence[str]]

_SHORT_NAME_PATTERN = re.compile(r'[A-Za-z0-9_-]{1,8}')

#: Axis names that are columns of the results table or the match key of a grid.
RESERVED_AXIS_NAMES = frozenset(
    {'grid', 'run_path', 'study_state', 'instance', 'instance_state', 'error'}
)

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
        ValueError: If the name is empty, starts with a dot (the hidden study
            folder is ``.study``), or holds any character outside
            ``[A-Za-z0-9._-]``.
    """
    if not name or name.startswith('.') or NAME_PART_PATTERN.fullmatch(name) is None:
        message = (
            f'invalid {kind} name {name!r}; names use letters, digits, dot, dash or underscore, '
            f'and do not start with a dot'
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

    Raises:
        ValueError: If two values have the same text, which would be one
            value name.
    """
    names = [str(value) for value in values]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        message = f'setting_axis({path!r}): the value {repeated[0]!r} is given twice'
        raise ValueError(message)
    return {str(value): {path: value} for value in values}


@dataclass(frozen=True, kw_only=True)
class Override:
    """Settings applied to the runs whose axis values match."""

    match: Match
    settings: Settings

    def __post_init__(self) -> None:
        """Check that match and settings are mappings."""
        for name in ('match', 'settings'):
            if not isinstance(getattr(self, name), Mapping):
                message = (
                    f'Override {name} must be a mapping, got {type(getattr(self, name)).__name__}'
                )
                raise TypeError(message)


@dataclass(frozen=True, kw_only=True)
class Grid:
    """One script command of a study, run over the product of its axes.

    Args:
        short_name: At most 8 characters, unique in the study; shown where a
            full grid name does not fit, e.g. on ``squeue``.
        command: The no-preset command of the script, as a word tuple.
        config: The config class of the script that the settings are typed against.
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
                f'use at most 8 letters, digits, dashes or underscores (squeue shows 8)'
            )
            raise ValueError(message)
        if not self.command:
            message = 'grid command must not be empty'
            raise ValueError(message)
        if not (isinstance(self.config, type) and dataclasses.is_dataclass(self.config)):
            message = f'grid config must be a dataclass, got {type(self.config).__qualname__}'
            raise ValueError(message)
        self._check_shapes()
        if not self.axes:
            message = 'grid must have at least one axis'
            raise ValueError(message)
        for axis_name, values in self.axes.items():
            validate_name(name=axis_name, kind='axis')
            if axis_name in RESERVED_AXIS_NAMES:
                message = (
                    f'axis name {axis_name!r} is reserved (a results column or the grid key); '
                    f'reserved: {", ".join(sorted(RESERVED_AXIS_NAMES))}'
                )
                raise ValueError(message)
            if not values:
                message = f'axis {axis_name!r} must have at least one value'
                raise ValueError(message)
            for value_name in values:
                validate_name(name=value_name, kind=f'axis {axis_name!r} value')

    def _check_shapes(self) -> None:
        """Type-check the containers; the usual slip is a missing trailing comma."""
        self._check_axis_shapes()
        if isinstance(self.exclude, Mapping):
            message = 'grid exclude must be a tuple of matches; trailing comma: exclude=({...},)?'
            raise TypeError(message)
        for match in self.exclude:
            if not isinstance(match, Mapping):
                message = f'each grid exclude must be a match mapping, got {_kind(value=match)}'
                raise TypeError(message)
        if isinstance(self.overrides, Override):
            message = (
                'grid overrides must be a tuple of Override; '
                'trailing comma: overrides=(Override(...),)?'
            )
            raise TypeError(message)
        for override in self.overrides:
            if not isinstance(override, Override):
                message = f'each grid override must be an Override, got {_kind(value=override)}'
                raise TypeError(message)
        if not isinstance(self.run_settings, Mapping):
            message = f'grid run_settings must be a mapping, got {_kind(value=self.run_settings)}'
            raise TypeError(message)

    def _check_axis_shapes(self) -> None:
        if not isinstance(self.axes, Mapping):
            message = (
                f'grid axes must be a mapping of axis name to values, got {_kind(value=self.axes)}'
            )
            raise TypeError(message)
        for axis_name, values in self.axes.items():
            if not isinstance(values, Mapping):
                message = (
                    f'axis {axis_name!r} must map value names to settings, '
                    f'got {_kind(value=values)}; for one setting use '
                    f'setting_axis(path=..., values=...)'
                )
                raise TypeError(message)
            for value_name, settings in values.items():
                if not isinstance(settings, Mapping):
                    message = (
                        f'axis {axis_name!r} value {value_name!r} must be a mapping of '
                        f'setting path to value, got {_kind(value=settings)}'
                    )
                    raise TypeError(message)


def _kind(*, value: object) -> str:
    return type(value).__name__


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
            if not isinstance(grid, Grid):
                message = f'grid {grid_name!r} must be a Grid, got {_kind(value=grid)}'
                raise TypeError(message)
            matches = (*grid.exclude, *(override.match for override in grid.overrides))
            problems = _grid_match_problems(grid_name=grid_name, grid=grid, matches=matches)
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
            if next(_grid_runs(grid_name=grid_name, grid=grid), None) is None:
                message = f'grid {grid_name!r} has no runs; every run is excluded'
                raise ValueError(message)

    def results(self, *, machine: Path) -> 'list[ResultRow]':
        """The rows of the results table of every run of the study.

        Args:
            machine: The machine profile the runs run on.

        Returns:
            One row per (run, instance), and one row per run without
            instances; the runs of every grid, in expansion order.
        """
        # why: imported here, not at the top, because the result table imports this module
        from tslab.study.result_table import study_results  # noqa: PLC0415

        return study_results(study=self, machine=machine)


def _grid_match_problems(*, grid_name: str, grid: Grid, matches: Sequence[Match]) -> list[str]:
    """The problems of the matches written inside one grid: a match there is about its own grid."""
    problems: list[str] = []
    for match in matches:
        normalized = normalize_match(match=match)
        foreign = [name for name in normalized.get('grid', ()) if name != grid_name]
        if foreign:
            problems.append(
                f'a match inside grid {grid_name!r} applies to its own grid only, '
                f'but names grid {", ".join(repr(name) for name in foreign)}'
            )
        rest = {key: value for key, value in normalized.items() if key != 'grid'}
        problems += match_problems(grids={grid_name: grid}, match=rest)
    return problems


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
        if not isinstance(key, str):
            message = f'match keys are axis names, got {key!r}'
            raise TypeError(message)
        if not isinstance(value, str | Sequence):
            message = f'match key {key!r} must be a name or a list of names, got {value!r}'
            raise TypeError(message)
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
                problems.append(
                    f'grid {", ".join(repr(name) for name in unknown)} is not a grid of this study'
                )
            continue
        axis_grids = [grid for grid in grids.values() if key in grid.axes]
        if not axis_grids:
            problems.append(f'{key} is not an axis of any grid')
            continue
        known = {name for grid in axis_grids for name in grid.axes[key]}
        unknown = [name for name in value if name not in known]
        if unknown:
            problems.append(f'{key} has no value {", ".join(repr(name) for name in unknown)}')
    return problems


def run_matches(*, grid_name: str, values: Mapping[str, str], match: Match) -> bool:
    """Whether a run of one grid matches: every key must hold.

    A key the grid of the run has no axis for makes the run not match.

    Args:
        grid_name: The grid the run belongs to.
        values: The axis name to value name of the run.
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


def merge_settings(
    *, base: Mapping[str, SettingValue], update: Mapping[str, SettingValue]
) -> dict[str, SettingValue]:
    """``update`` over ``base`` path by path; a changed choice drops the old children.

    When ``update`` gives a path a different value than ``base`` has, the
    paths of ``base`` below it are dropped unless ``update`` sets them too:
    switching ``model`` from ``'ts2vec'`` to ``'cost'`` must not keep
    ``model.depth``, which ``cost`` does not have.

    Args:
        base: The settings to start from.
        update: The settings applied on top.

    Returns:
        The merged settings.
    """
    result = dict(base)
    for path, value in update.items():
        if path in result and result[path] != value:
            prefix = f'{path}.'
            for stale in [key for key in result if key.startswith(prefix) and key not in update]:
                del result[stale]
        result[path] = value
    return result


def matching_in_order(
    *, matches: Sequence[Match], grid_name: str, values: Mapping[str, str]
) -> list[int]:
    """The positions of the matches that hold for a run, in precedence order.

    More match keys come later (win); between equal ones the earlier position
    comes first, so the later is applied last.

    Args:
        matches: The matches, e.g. of overrides.
        grid_name: The grid the run belongs to.
        values: The axis name to value name of the run.

    Returns:
        The positions into ``matches`` to apply, first to last.
    """
    matching = [
        position
        for position, match in enumerate(matches)
        if run_matches(grid_name=grid_name, values=values, match=match)
    ]
    matching.sort(key=lambda position: len(matches[position]))
    return matching


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
    later is applied last. Settings merge as in ``merge_settings``.

    Args:
        base: The settings to apply the overrides to.
        overrides: Pairs of match and settings.
        grid_name: The grid the run belongs to.
        values: The axis name to value name of the run.

    Returns:
        The base settings with the matching overrides applied in order.
    """
    result = dict(base)
    order = matching_in_order(
        matches=[match for match, _ in overrides], grid_name=grid_name, values=values
    )
    for position in order:
        result = merge_settings(base=result, update=overrides[position][1])
    return result


def _grid_runs(*, grid_name: str, grid: Grid) -> Iterator[dict[str, str]]:
    """The axis values of every run of the grid that is not excluded, in axis order."""
    axis_names = list(grid.axes)
    for combo in product(*(grid.axes[name] for name in axis_names)):
        values = dict(zip(axis_names, combo, strict=True))
        if not any(
            run_matches(grid_name=grid_name, values=values, match=match) for match in grid.exclude
        ):
            yield values


def expand_runs(*, study: Study) -> list[StudyRun]:
    """Expand a study into its runs, grids in order and axes in order.

    Per grid the runs are the product of its axes; excluded runs are dropped.

    Args:
        study: The study to expand.

    Returns:
        The runs, one per point of the axis product of every grid that is not
        excluded, in grid order and then axis order.
    """
    runs: list[StudyRun] = []
    for grid_name, grid in study.grids.items():
        overrides = [(override.match, override.settings) for override in grid.overrides]
        for values in _grid_runs(grid_name=grid_name, grid=grid):
            settings: dict[str, SettingValue] = {}
            for name in grid.axes:
                settings = merge_settings(base=settings, update=grid.axes[name][values[name]])
            settings = merge_settings(base=settings, update=grid.run_settings)
            settings = merge_overrides(
                base=settings, overrides=overrides, grid_name=grid_name, values=values
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
    grid has or a value the axis of no grid has raises. Filtering never changes
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
            an axis belongs to no grid or a value belongs to the axis of no grid.
    """
    unknown_grids = [name for name in grids if name not in study.grids]
    if unknown_grids:
        message = (
            f'unknown grid {", ".join(repr(name) for name in unknown_grids)}; '
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
    """One ``AXIS=V1,V2`` item, checked against the grids of the study."""
    axis, separator, values = item.partition('=')
    if not separator or not axis or not values:
        message = (
            f'only item {item!r} is malformed; items are an axis name as AXIS=V1,V2, '
            'where every V names a value of that axis'
        )
        raise ValueError(message)
    value_names = tuple(values.split(','))
    if any('=' in name for name in value_names):
        message = (
            f'only item {item!r} names more than one axis; use one --only per axis, '
            'as --only recipe=b1 --only dataset=Coffee'
        )
        raise ValueError(message)
    problems = match_problems(grids=study.grids, match={axis: value_names})
    if problems:
        raise ValueError('; '.join(problems))
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

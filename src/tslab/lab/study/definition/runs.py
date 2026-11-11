"""Runs: the expansion of a study into runs, and the selection of runs by grid and axis values."""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import TYPE_CHECKING

from tslab.lab.study.definition.grid import DEFAULT_RUN_SETTINGS, Grid
from tslab.lab.study.definition.matching import Match, match_problems, run_matches
from tslab.lab.study.definition.settings import check_paths, Settings, SettingValue

if TYPE_CHECKING:
    from tslab.lab.study.definition.study import Study

__all__ = [
    'StudyRun',
    'expand_runs',
    'grid_runs',
    'matching_in_order',
    'merge_overrides',
    'merge_settings',
    'select_runs',
]


@dataclass(frozen=True, kw_only=True)
class StudyRun:
    """One run of a grid: its axis values, run path and merged settings.

    Attributes:
        grid: The name of the grid the run belongs to.
        values: The axis name to value name of the run; read-only.
        run_path: The run path within the study.
        settings: The merged settings of the run, path to value; read-only.
    """

    grid: str
    values: Mapping[str, str]
    run_path: PurePosixPath
    settings: Mapping[str, SettingValue]

    def __post_init__(self) -> None:
        """Copy the mappings read-only, so a frozen run cannot change through them."""
        object.__setattr__(self, 'values', MappingProxyType(dict(self.values)))
        object.__setattr__(self, 'settings', MappingProxyType(dict(self.settings)))


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
            _drop_children(result=result, path=path, keep=update)
        result[path] = value
    return result


def _drop_children(
    *, result: dict[str, SettingValue], path: str, keep: Mapping[str, SettingValue]
) -> None:
    """Remove the paths below ``path`` from ``result`` unless ``keep`` sets them."""
    prefix = f'{path}.'
    for stale in [key for key in result if key.startswith(prefix) and key not in keep]:
        del result[stale]


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


def grid_runs(*, grid_name: str, grid: Grid) -> Iterator[dict[str, str]]:
    """The axis values of every run of the grid that is not excluded, in axis order."""
    axis_names = list(grid.axes)
    for combo in product(*(grid.axes[name] for name in axis_names)):
        values = dict(zip(axis_names, combo, strict=True))
        if not any(
            run_matches(grid_name=grid_name, values=values, match=match) for match in grid.exclude
        ):
            yield values


def expand_runs(*, study: 'Study') -> list[StudyRun]:
    """Expand a study into its runs, grids in order and axes in order.

    Per grid the runs are the product of its axes; excluded runs are dropped.

    Args:
        study: The study to expand.

    Returns:
        The runs, one per point of the axis product of every grid that is not
        excluded, in grid order and then axis order.
    """
    return [
        _study_run(grid_name=grid_name, grid=grid, values=values, study_settings=study.settings)
        for grid_name, grid in study.grids.items()
        for values in grid_runs(grid_name=grid_name, grid=grid)
    ]


def _study_run(
    *, grid_name: str, grid: Grid, values: Mapping[str, str], study_settings: Settings
) -> StudyRun:
    """The run of one point of the axis product of a grid, with its merged settings."""
    overrides = [(override.match, override.settings) for override in grid.overrides]
    settings: dict[str, SettingValue] = {}
    for name in grid.axes:
        settings = merge_settings(base=settings, update=grid.axes[name][values[name]])
    settings = merge_settings(
        base=settings,
        update={
            path: value
            for path, value in study_settings.items()
            if not check_paths(config_class=grid.config, settings={path: value})
        },
    )
    settings = merge_settings(base=settings, update=DEFAULT_RUN_SETTINGS)
    settings = merge_settings(base=settings, update=grid.run_settings)
    settings = merge_overrides(
        base=settings, overrides=overrides, grid_name=grid_name, values=values
    )
    return StudyRun(
        grid=grid_name,
        values=dict(values),
        run_path=grid.run_path_for(grid=grid_name, values=values),
        settings=settings,
    )


def select_runs(
    *, study: 'Study', runs: Sequence[StudyRun], grids: Sequence[str] = (), only: Sequence[str] = ()
) -> list[StudyRun]:
    """The runs that pass the grid names and axis filters.

    An empty ``grids`` keeps every grid; a name that is not a grid raises.
    Each ``only`` item is ``AXIS=V1,V2``: among the grids that have ``AXIS`` it
    keeps the runs whose value for it is named, and it leaves the runs of the
    grids without ``AXIS`` alone (unlike ``run_matches``, where a missing axis
    never matches, because a filter must not empty a study that mixes grids).
    A malformed item, an axis no grid has or a value the axis of no grid has
    raises. Filtering never changes run paths.

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


def _parse_only(*, item: str, study: 'Study') -> tuple[str, tuple[str, ...]]:
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

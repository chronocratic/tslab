"""Dependencies between grids of a study."""

from collections.abc import Mapping
from dataclasses import dataclass

from tslab.lab.enums import AfterCondition, Satisfy
from tslab.lab.study.definition.definition import expand_runs, Grid, Study, StudyRun

__all__ = ['After', 'after_problems', 'chained_grids', 'grid_order', 'match_axes', 'upstream_runs']


@dataclass(frozen=True, kw_only=True)
class After:
    """An upstream grid and the condition its runs must meet.

    Attributes:
        grid: The upstream grid name.
        match: Axes used to pair runs; ``None`` uses shared axes and ``()`` fans out.
        condition: The condition each matched upstream run must meet.
        delay_minutes: Minutes after an upstream run starts; only used with ``STARTED``.
        satisfy: Whether all or any matched upstream runs must meet the condition.
    """

    grid: str
    match: tuple[str, ...] | None = None
    condition: AfterCondition = AfterCondition.OK
    delay_minutes: int = 0
    satisfy: Satisfy = Satisfy.ALL

    def __post_init__(self) -> None:
        """Check that delay is non-negative and only used with ``STARTED``."""
        if self.delay_minutes < 0:
            message = 'delay_minutes must be at least 0'
            raise ValueError(message)
        if self.delay_minutes and self.condition is not AfterCondition.STARTED:
            message = "delay_minutes needs condition 'started'"
            raise ValueError(message)


def match_axes(*, study: Study, grid_name: str, after: After) -> tuple[str, ...]:
    """The axes used to pair one downstream run with upstream runs.

    Args:
        study: The study containing both grids.
        grid_name: The downstream grid name.
        after: The dependency declared by the downstream grid.

    Returns:
        Explicit axes, or shared axes in downstream order.
    """
    if after.match is not None:
        return after.match
    upstream_axes = study.grids[after.grid].axes
    return tuple(axis for axis in study.grids[grid_name].axes if axis in upstream_axes)


def upstream_runs(*, study: Study, run: StudyRun, after: After) -> tuple[StudyRun, ...]:
    """The upstream runs paired with one downstream run, in expansion order.

    Args:
        study: The study containing the runs.
        run: The downstream run.
        after: The dependency whose upstream runs are selected.

    Returns:
        Matching runs of ``after.grid``.
    """
    axes = match_axes(study=study, grid_name=run.grid, after=after)
    expanded = expand_runs(study=study)
    candidates = tuple(candidate for candidate in expanded if candidate.grid == after.grid)
    return _matching_runs(downstream=run, upstream=candidates, axes=axes)


def chained_grids(*, study: Study) -> frozenset[str]:
    """The grids that declare or are named by a dependency.

    Args:
        study: The study to inspect.

    Returns:
        Grid names participating in at least one dependency.
    """
    downstream = {name for name, grid in study.grids.items() if grid.after}
    upstream = {after.grid for grid in study.grids.values() for after in grid.after}
    return frozenset(downstream | upstream)


def grid_order(*, study: Study) -> tuple[str, ...]:
    """The topological order of grids, with ties in study order.

    Args:
        study: The study to order.

    Returns:
        Grid names with upstream grids before their dependents.
    """
    remaining = set(study.grids)
    ordered: list[str] = []
    while remaining:
        ready = next(
            (
                name
                for name in study.grids
                if name in remaining
                and all(after.grid not in remaining for after in study.grids[name].after)
            ),
            None,
        )
        if ready is None:
            return tuple(ordered)
        ordered.append(ready)
        remaining.remove(ready)
    return tuple(ordered)


def after_problems(*, study: Study) -> list[str]:
    """The dependency problems found while validating a study.

    Args:
        study: The study to validate.

    Returns:
        One message per invalid dependency or unmatched downstream run.
    """
    expanded = expand_runs(study=study)
    runs_by_grid = {
        name: tuple(run for run in expanded if run.grid == name) for name in study.grids
    }
    problems = [
        problem
        for grid_name, grid in study.grids.items()
        for problem in _grid_after_problems(
            study=study, grid_name=grid_name, grid=grid, runs_by_grid=runs_by_grid
        )
    ]
    cycle = _first_cycle(study=study)
    if cycle:
        problems.append(f'after cycle: {" -> ".join(cycle)}')
    return problems


def _grid_after_problems(
    *, study: Study, grid_name: str, grid: Grid, runs_by_grid: Mapping[str, tuple[StudyRun, ...]]
) -> list[str]:
    """The dependency problems declared by one downstream grid."""
    problems = []
    if len(grid.after) > 1 and any(after.satisfy is Satisfy.ANY for after in grid.after):
        problems.append(f"grid {grid_name!r}: satisfy 'any' needs a single after entry")
    for after in grid.after:
        problems.extend(
            _after_entry_problems(
                study=study, grid_name=grid_name, grid=grid, after=after, runs_by_grid=runs_by_grid
            )
        )
    return problems


def _after_entry_problems(
    *,
    study: Study,
    grid_name: str,
    grid: Grid,
    after: After,
    runs_by_grid: Mapping[str, tuple[StudyRun, ...]],
) -> list[str]:
    """The validation problems of one dependency entry."""
    if after.grid not in study.grids:
        return [f'grid {grid_name!r}: after names unknown grid {after.grid!r}']
    own_grid_problem = (
        [f'grid {grid_name!r}: after names its own grid'] if after.grid == grid_name else []
    )
    upstream = study.grids[after.grid]
    axes = match_axes(study=study, grid_name=grid_name, after=after)
    invalid_axes = _invalid_match_axes(
        grid_name=grid_name,
        upstream_name=after.grid,
        axes=axes,
        downstream_axes=grid.axes,
        upstream_axes=upstream.axes,
    )
    if invalid_axes:
        return [*own_grid_problem, *invalid_axes]
    unmatched = [
        f'run {run.run_path} matches no run of grid {after.grid!r}'
        for run in runs_by_grid[grid_name]
        if not _matching_runs(downstream=run, upstream=runs_by_grid[after.grid], axes=axes)
    ]
    return [*own_grid_problem, *unmatched]


def _invalid_match_axes(
    *,
    grid_name: str,
    upstream_name: str,
    axes: tuple[str, ...],
    downstream_axes: Mapping[str, object],
    upstream_axes: Mapping[str, object],
) -> list[str]:
    """The explicit match axes missing from either grid."""
    return [
        f'grid {grid_name!r}: after match axis {axis!r} is not an axis of grid {grid!r}'
        for axis in axes
        for grid, grid_axes in ((grid_name, downstream_axes), (upstream_name, upstream_axes))
        if axis not in grid_axes
    ]


def _matching_runs(
    *, downstream: StudyRun, upstream: tuple[StudyRun, ...], axes: tuple[str, ...]
) -> tuple[StudyRun, ...]:
    """The upstream runs with equal values on all pairing axes."""
    return tuple(
        candidate
        for candidate in upstream
        if all(candidate.values.get(axis) == downstream.values.get(axis) for axis in axes)
    )


def _first_cycle(*, study: Study) -> tuple[str, ...]:
    """The first dependency cycle in study and declaration order."""
    visited = frozenset[str]()
    for grid_name in study.grids:
        cycle, visited = _visit_grid(study=study, grid_name=grid_name, visited=visited, path=())
        if cycle:
            return cycle
    return ()


def _visit_grid(
    *, study: Study, grid_name: str, visited: frozenset[str], path: tuple[str, ...]
) -> tuple[tuple[str, ...], frozenset[str]]:
    """Depth-first visit one grid, returning its cycle and newly visited grids."""
    if grid_name in path:
        return (*path[path.index(grid_name) :], grid_name), visited
    if grid_name in visited:
        return (), visited
    next_path = (*path, grid_name)
    for after in study.grids[grid_name].after:
        if after.grid not in study.grids:
            continue
        cycle, visited = _visit_grid(
            study=study, grid_name=after.grid, visited=visited, path=next_path
        )
        if cycle:
            return cycle, visited
    return (), visited | {grid_name}

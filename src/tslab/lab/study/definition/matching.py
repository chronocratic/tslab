"""Match mappings: select the runs of a grid by axis values."""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tslab.lab.study.definition.grid import Grid

__all__ = ['Match', 'match_problems', 'normalize_match', 'run_matches']


type Match = Mapping[str, str | Sequence[str]]


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
    return {key: _normalized_names(key=key, value=value) for key, value in match.items()}


def _normalized_names(*, key: object, value: str | Sequence[str]) -> tuple[str, ...]:
    """The value names of one match key as a tuple, checked."""
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
    return values


def match_problems(*, grids: Mapping[str, 'Grid'], match: Match) -> list[str]:
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
        problems += _key_problems(grids=grids, key=key, value=value)
    return problems


def _key_problems(*, grids: Mapping[str, 'Grid'], key: str, value: tuple[str, ...]) -> list[str]:
    """The problems of one match key against the grids."""
    if key == 'grid':
        unknown = [name for name in value if name not in grids]
        names = ', '.join(repr(name) for name in unknown)
        return [f'grid {names} is not a grid of this study'] if unknown else []
    axis_grids = [grid for grid in grids.values() if key in grid.axes]
    if not axis_grids:
        return [f'{key} is not an axis of any grid']
    known = {name for grid in axis_grids for name in grid.axes[key]}
    unknown = [name for name in value if name not in known]
    names = ', '.join(repr(name) for name in unknown)
    return [f'{key} has no value {names}'] if unknown else []


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

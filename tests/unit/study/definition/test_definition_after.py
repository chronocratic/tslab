"""Grid dependencies: declaration, axis matching, validation and ordering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import cast, TYPE_CHECKING

import pytest

from tslab import enums
import tslab.study as study_api
from tslab.study import Grid, setting_axis, Study
import tslab.study.definition as definition_api
from tslab.study.definition import expand_runs, Settings, StudyRun
from tslab.study.definition.after import _visit_grid

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from tslab.study import After


@dataclass(frozen=True, kw_only=True)
class ChainConfig:
    """The minimal config for dependency declaration tests."""

    seed: int = 0
    run_path: Path = Path('chain')
    output_root: Path = Path('runs')


def _enum_value(*, name: str, value: str) -> object:
    assert hasattr(enums, name), f'{name} is not defined'
    enum_type = cast('Callable[[str], object]', getattr(enums, name))
    return enum_type(value)


def _after(
    *,
    grid: str,
    match: tuple[str, ...] | None = None,
    condition: str = 'ok',
    delay_minutes: int = 0,
    satisfy: str = 'all',
) -> After:
    assert hasattr(study_api, 'After'), 'After is not defined'
    after_type = cast('Callable[..., object]', study_api.After)
    return cast(
        'After',
        after_type(
            grid=grid,
            match=match,
            condition=_enum_value(name='AfterCondition', value=condition),
            delay_minutes=delay_minutes,
            satisfy=_enum_value(name='Satisfy', value=satisfy),
        ),
    )


def _axes(*, datasets: tuple[str, ...], head: bool = False) -> dict[str, Mapping[str, Settings]]:
    axes: dict[str, Mapping[str, Settings]] = {
        'dataset': {name: {} for name in datasets},
        'seed': setting_axis(path='seed', values=(0, 1)),
    }
    if head:
        axes['head'] = {'svm': {}}
    return axes


def _grid(
    *, short_name: str, axes: dict[str, Mapping[str, Settings]], after: tuple[After, ...] = ()
) -> Grid:
    run_path = '{grid}/{dataset}/seed_{seed}'
    if 'head' in axes:
        run_path += '/{head}'
    return Grid(
        short_name=short_name,
        command=('python', '-m', 'example'),
        config=ChainConfig,
        axes=axes,
        after=after,
        run_path=run_path,
    )


def chain_study(
    *,
    probe_datasets: tuple[str, ...] = ('a', 'b'),
    probe_after: tuple[After, ...] | None = None,
    reverse_grid_order: bool = False,
    include_other: bool = False,
) -> Study:
    """The two-grid chain from the tests, with two datasets and two seeds."""
    if probe_after is None:
        probe_after = (_after(grid='pretrain'),)
    pretrain = _grid(short_name='pre', axes=_axes(datasets=('a', 'b')))
    probe = _grid(
        short_name='probe', axes=_axes(datasets=probe_datasets, head=True), after=probe_after
    )
    grids = {'pretrain': pretrain, 'probe': probe}
    if include_other:
        other = _grid(short_name='other', axes=_axes(datasets=('a',)))
        grids = {'pretrain': pretrain, 'other': other, 'probe': probe}
    if reverse_grid_order:
        grids = {'probe': probe, **{name: grid for name, grid in grids.items() if name != 'probe'}}
    return Study(name='chain', grids=grids)


def run_of(*, study: Study, grid_name: str, values: Mapping[str, str]) -> StudyRun:
    return next(
        run for run in expand_runs(study=study) if run.grid == grid_name and run.values == values
    )


def _definition_function(*, name: str) -> Callable[..., object]:
    assert hasattr(definition_api, name), f'{name} is not defined'
    return cast('Callable[..., object]', getattr(definition_api, name))


def test_after_defaults() -> None:
    after = _after(grid='pretrain')

    assert (after.match, after.condition, after.delay_minutes, after.satisfy) == (
        None,
        _enum_value(name='AfterCondition', value='ok'),
        0,
        _enum_value(name='Satisfy', value='all'),
    )


def test_after_condition_enum_parses_all_values() -> None:
    assert [
        _enum_value(name='AfterCondition', value=value)
        for value in ('ok', 'any', 'failed', 'started')
    ] == [
        enums.AfterCondition.OK,
        enums.AfterCondition.ANY,
        enums.AfterCondition.FAILED,
        enums.AfterCondition.STARTED,
    ]


def test_satisfy_enum_parses_all_values() -> None:
    assert [_enum_value(name='Satisfy', value=value) for value in ('all', 'any')] == [
        enums.Satisfy.ALL,
        enums.Satisfy.ANY,
    ]


def test_matched_runs_pair_on_shared_axes() -> None:
    study = chain_study()
    probe = run_of(
        study=study, grid_name='probe', values={'dataset': 'a', 'seed': '0', 'head': 'svm'}
    )
    after = study.grids['probe'].after[0]
    upstream_runs = cast(
        'Callable[..., tuple[StudyRun, ...]]', _definition_function(name='upstream_runs')
    )

    assert [
        run.run_path.as_posix() for run in upstream_runs(study=study, run=probe, after=after)
    ] == ['pretrain/a/seed_0']


def test_empty_match_takes_every_upstream_run() -> None:
    study = chain_study(probe_after=(_after(grid='pretrain', match=()),))
    probe = run_of(
        study=study, grid_name='probe', values={'dataset': 'a', 'seed': '0', 'head': 'svm'}
    )
    after = study.grids['probe'].after[0]
    upstream_runs = cast(
        'Callable[..., tuple[StudyRun, ...]]', _definition_function(name='upstream_runs')
    )

    assert len(upstream_runs(study=study, run=probe, after=after)) == 4


def test_match_axes_keeps_downstream_order_or_explicit_order() -> None:
    study = chain_study()
    match_axes = cast('Callable[..., tuple[str, ...]]', _definition_function(name='match_axes'))

    assert match_axes(study=study, grid_name='probe', after=_after(grid='pretrain')) == (
        'dataset',
        'seed',
    )
    assert match_axes(
        study=study, grid_name='probe', after=_after(grid='pretrain', match=('seed', 'dataset'))
    ) == ('seed', 'dataset')


def test_chained_grids_includes_downstream_and_upstream() -> None:
    chained_grids = cast(
        'Callable[..., frozenset[str]]', _definition_function(name='chained_grids')
    )

    assert chained_grids(study=chain_study()) == frozenset({'pretrain', 'probe'})


def test_grid_order_puts_upstream_first() -> None:
    study = chain_study(reverse_grid_order=True)
    grid_order = cast('Callable[..., tuple[str, ...]]', _definition_function(name='grid_order'))

    assert grid_order(study=study) == ('pretrain', 'probe')


def test_grid_order_uses_study_order_for_available_grids() -> None:
    study = chain_study(reverse_grid_order=True, include_other=True)
    grid_order = cast('Callable[..., tuple[str, ...]]', _definition_function(name='grid_order'))

    assert grid_order(study=study) == ('pretrain', 'probe', 'other')


def test_grid_after_requires_a_tuple() -> None:
    with pytest.raises(TypeError, match='trailing comma'):
        _grid(
            short_name='probe',
            axes=_axes(datasets=('a',)),
            after=cast('tuple[After, ...]', _after(grid='pretrain')),
        )


def test_grid_after_requires_after_entries() -> None:
    with pytest.raises(TypeError, match='each grid after must be an After'):
        _grid(
            short_name='probe',
            axes=_axes(datasets=('a',)),
            after=cast('tuple[After, ...]', ('pretrain',)),
        )


def test_invalid_after_reference_is_rejected() -> None:
    with pytest.raises(
        ValueError, match=re.escape("grid 'probe': after names unknown grid 'nope'")
    ):
        chain_study(probe_after=(_after(grid='nope'),))


def test_after_own_grid_is_rejected() -> None:
    with pytest.raises(ValueError, match=re.escape("grid 'probe': after names its own grid")):
        chain_study(probe_after=(_after(grid='probe'),))


def test_match_axis_missing_upstream_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape("grid 'probe': after match axis 'head' is not an axis of grid 'pretrain'"),
    ):
        chain_study(probe_after=(_after(grid='pretrain', match=('head',)),))


def test_delay_requires_started_condition() -> None:
    with pytest.raises(ValueError, match=re.escape("delay_minutes needs condition 'started'")):
        _after(grid='pretrain', delay_minutes=5)


def test_delay_minutes_must_not_be_negative() -> None:
    with pytest.raises(ValueError, match='delay_minutes must be at least 0'):
        _after(grid='pretrain', condition='started', delay_minutes=-1)


def test_satisfy_any_needs_one_after_entry() -> None:
    after = (_after(grid='pretrain', satisfy='any'), _after(grid='other'))
    with pytest.raises(ValueError, match=re.escape("satisfy 'any' needs a single after entry")):
        chain_study(probe_after=after, include_other=True)


def test_after_cycle_is_rejected() -> None:
    with pytest.raises(ValueError, match=re.escape('after cycle: a -> b -> a')):
        Study(
            name='cycle',
            grids={
                'a': _grid(short_name='a', axes=_axes(datasets=('a',)), after=(_after(grid='b'),)),
                'b': _grid(short_name='b', axes=_axes(datasets=('a',)), after=(_after(grid='a'),)),
            },
        )


def test_visit_grid_returns_new_visited_state() -> None:
    study = chain_study()
    visited = frozenset()

    cycle, updated = _visit_grid(study=study, grid_name='pretrain', visited=visited, path=())

    assert cycle == ()
    assert updated == frozenset({'pretrain'})
    assert visited == frozenset()


def test_downstream_run_without_upstream_match_is_rejected() -> None:
    with pytest.raises(
        ValueError, match=re.escape("run probe/c/seed_0/svm matches no run of grid 'pretrain'")
    ):
        chain_study(probe_datasets=('a', 'c'))

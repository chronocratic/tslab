"""Matches: exclude and override rules, checked at study build time."""

import pytest

from study.configs import grid
from tslab.study import match_problems, Override, run_matches, Study


def test_run_matches_lists_and_grid_key() -> None:
    values = {'recipe': 'ts2vec-paper', 'dataset': 'Coffee', 'seed': '42'}
    assert run_matches(
        grid_name='g', values=values, match={'recipe': ['cost-paper', 'ts2vec-paper']}
    )
    assert run_matches(grid_name='g', values=values, match={'grid': 'g', 'dataset': 'Coffee'})
    assert not run_matches(grid_name='g', values=values, match={'grid': 'h'})
    assert not run_matches(grid_name='g', values=values, match={'variant': 'ETTh1'})


def test_unknown_value_name_in_match_raises() -> None:
    with pytest.raises(ValueError, match='Cofee'):
        Study(name='s', grids={'g': grid(exclude=({'dataset': 'Cofee'},))})


def test_unknown_axis_in_match_raises() -> None:
    with pytest.raises(ValueError, match='datset'):
        Study(
            name='s',
            grids={'g': grid(overrides=(Override(match={'datset': 'Coffee'}, settings={}),))},
        )


def test_match_problems_across_grids() -> None:
    grids = {'c': grid(), 'f': grid(axes={'variant': {'ETTh1': {}}})}
    assert match_problems(grids=grids, match={'variant': 'ETTh1', 'grid': 'f'}) == []
    assert len(match_problems(grids=grids, match={'variant': 'ETTh9', 'grid': 'x'})) == 2

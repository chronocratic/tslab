"""``expand_runs``: a study into its runs, with merged settings."""

import pytest

from tests.unit.study.configs import grid
from tslab.study import Override, Study
from tslab.study.definition import expand_runs


def test_run_path_follows_axis_order() -> None:
    runs = expand_runs(study=Study(name='s', grids={'classification': grid()}))
    assert [run.run_path for run in runs][:2] == [
        'classification/ts2vec-paper/Coffee/42',
        'classification/ts2vec-paper/Coffee/7',
    ]
    assert len(runs) == 4


def test_merge_order_axes_run_settings_overrides() -> None:
    g = grid(overrides=(Override(match={'dataset': 'Coffee'}, settings={'seed': 0}),))
    run = expand_runs(study=Study(name='s', grids={'g': g}))[0]
    assert run.settings['seed'] == 0
    assert run.settings['output_root'] == '{output_root}/{study}'
    assert run.settings['model'] == 'ts2vec'


def test_override_precedence() -> None:
    g = grid(
        overrides=(
            Override(
                match={'recipe': 'ts2vec-paper', 'dataset': 'Coffee'},
                settings={'data.batch_size': 2},
            ),
            Override(match={'dataset': 'Coffee'}, settings={'data.batch_size': 4}),
            Override(match={'dataset': 'Coffee'}, settings={'data.batch_size': 3}),
        )
    )
    by_path = {run.run_path: run for run in expand_runs(study=Study(name='s', grids={'g': g}))}
    assert by_path['g/ts2vec-paper/Coffee/42'].settings['data.batch_size'] == 2
    assert by_path['g/cost-paper/Coffee/42'].settings['data.batch_size'] == 3


def test_exclude_drops_runs() -> None:
    g = grid(exclude=({'recipe': 'cost-paper', 'seed': ['7']},))
    assert len(expand_runs(study=Study(name='s', grids={'g': g}))) == 3


def test_unknown_path_names_grid_value_and_path() -> None:
    bad = grid(axes={'seed': {'42': {'sed': 42}}})
    with pytest.raises(ValueError, match=r"grid 'g'.*seed.*42.*sed"):
        Study(name='s', grids={'g': bad})

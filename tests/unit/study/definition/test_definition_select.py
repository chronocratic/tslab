"""``select_runs``: filter the expanded runs by grid name and axis values."""

import pytest

from tests.unit.study.configs import grid
from tslab.study import setting_axis, Study
from tslab.study.definition import expand_runs, select_runs


def test_only_filters_grids_with_that_axis() -> None:
    study = Study(
        name='s',
        grids={
            'c': grid(),
            'f': grid(
                short_name='fc',
                axes={'variant': {'ETTh1': {}}, 'seed': setting_axis(path='seed', values=(42,))},
            ),
        },
    )
    runs = expand_runs(study=study)
    selected = select_runs(study=study, runs=runs, only=('dataset=Coffee', 'seed=42'))
    assert {run.run_path for run in selected} == {
        'c/ts2vec-paper/Coffee/42',
        'c/cost-paper/Coffee/42',
        'f/ETTh1/42',
    }
    assert [run.run_path for run in select_runs(study=study, runs=runs, grids=('f',))] == [
        'f/ETTh1/42'
    ]


@pytest.mark.parametrize('item', ['datset=Coffee', 'dataset=Cofee', 'dataset', 'dataset='])
def test_only_with_unknown_axis_or_value_raises(item: str) -> None:
    study = Study(name='s', grids={'c': grid()})
    with pytest.raises(ValueError, match=r'axis|has no value'):
        select_runs(study=study, runs=expand_runs(study=study), only=(item,))


def test_only_with_two_axes_in_one_item_says_to_repeat_the_flag() -> None:
    study = Study(name='s', grids={'c': grid()})
    with pytest.raises(ValueError, match='one --only per axis'):
        select_runs(study=study, runs=expand_runs(study=study), only=('dataset=Coffee,seed=42',))

"""``Grid.run_path``: template expansion and the checks at study load."""

from pathlib import PurePosixPath
import re
from time import perf_counter

import pytest

from tests.unit.study.configs import grid
from tslab.study import Study
from tslab.study.definition import expand_runs


def test_default_template_is_grid_then_axes() -> None:
    assert grid().run_path_template == '{grid}/{recipe}/{dataset}/{seed}'


def test_default_paths_unchanged() -> None:
    runs = expand_runs(study=Study(name='s', grids={'classification': grid()}))
    assert runs[0].run_path == PurePosixPath('classification/ts2vec-paper/Coffee/42')


def test_template_labels_seed_and_keeps_value_plain() -> None:
    g = grid(run_path='{grid}/{dataset}/{recipe}/seed_{seed}')
    runs = expand_runs(study=Study(name='s', grids={'cls': g}))
    assert runs[0].run_path == PurePosixPath('cls/Coffee/ts2vec-paper/seed_42')
    assert runs[0].values['seed'] == '42'


def test_short_name_placeholder() -> None:
    g = grid(run_path='{short_name}-{recipe}-{dataset}-s{seed}')
    values = {'recipe': 'r', 'dataset': 'd', 'seed': '7'}
    assert g.run_path_for(grid='x', values=values) == PurePosixPath('cls-r-d-s7')


@pytest.mark.parametrize(
    ('template', 'needle'),
    [
        ('{model}/{seed}', "unknown placeholder 'model'"),
        ('{study}/{seed}', "unknown placeholder 'study'"),
        ('{{seed}}', 'not allowed'),
        ('{grid} {seed}', 'not allowed'),
        ('${HOME}/{seed}', 'not allowed'),
    ],
)
def test_bad_run_path_templates_fail(template: str, needle: str) -> None:
    with pytest.raises(ValueError, match=re.escape(needle)):
        grid(run_path=template)


@pytest.mark.parametrize(
    'template',
    [
        '{grid}//{recipe}/{dataset}/{seed}',
        '/{grid}/{recipe}/{dataset}/{seed}',
        '{grid}/{recipe}/{dataset}/{seed}/',
        '{grid}/{recipe}/{dataset}/.{seed}',
    ],
)
def test_bad_run_path_parts_fail(template: str) -> None:
    with pytest.raises(ValueError, match=r"grid 'a'.*run path"):
        Study(name='s', grids={'a': grid(run_path=template)})


def test_template_missing_an_axis_with_two_values_fails() -> None:
    with pytest.raises(ValueError, match=r"run path 'a/ts2vec-paper/Coffee'.*seed"):
        Study(name='s', grids={'a': grid(run_path='{grid}/{recipe}/{dataset}')})


def test_run_paths_must_differ_across_grids() -> None:
    template = '{recipe}/{dataset}/{seed}'
    with pytest.raises(ValueError, match=r"grid 'b'.*grid 'a'"):
        Study(
            name='s',
            grids={
                'a': grid(short_name='a', run_path=template),
                'b': grid(short_name='b', run_path=template),
            },
        )


def test_axis_with_one_value_may_be_left_out() -> None:
    Study(name='s', grids={'a': grid(run_path='{grid}/{recipe}/{seed}')})


def test_run_path_inside_another_run_path_fails() -> None:
    with pytest.raises(ValueError, match=r"grid 'b'.*inside or around"):
        Study(
            name='s',
            grids={
                'a': grid(short_name='a'),
                'b': grid(short_name='b', run_path='a/{recipe}/{dataset}'),
            },
        )


def test_run_paths_that_differ_only_in_case_fail() -> None:
    axes = {'a': {'Coffee': {}, 'coffee': {}}}
    with pytest.raises(ValueError, match='run path'):
        Study(name='s', grids={'g': grid(axes=axes)})


def test_run_path_part_with_temporary_marker_fails() -> None:
    with pytest.raises(ValueError, match=r'\.tmp-'):
        Study(name='s', grids={'a': grid(run_path='{grid}/{recipe}/{dataset}/x.tmp-{seed}')})


def test_study_checks_many_run_paths_without_quadratic_scan() -> None:
    values = {str(index): {} for index in range(8_000)}
    study_grid = grid(axes={'replicate': values})

    started = perf_counter()
    Study(name='s', grids={'g': study_grid})

    assert perf_counter() - started < 3.0

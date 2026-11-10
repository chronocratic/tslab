"""``Grid.job_name``: template expansion and the checks at study load."""

import re

import pytest

from tests.unit.study.configs import grid
from tslab.study import Study
from tslab.study.definition import study_job_names


def test_default_job_name_is_study_and_short_name() -> None:
    g = grid()
    assert g.job_name == '{study}_{short_name}'
    assert g.job_name_axes == ()
    assert g.job_name_for(study='s', values={'recipe': 'cost-paper', 'seed': '7'}) == 's_cls'


def test_empty_job_name_template_fails() -> None:
    with pytest.raises(ValueError, match=r'job_name.*must not be empty'):
        grid(job_name='')


def test_job_name_template_expands_axes_in_template_order() -> None:
    g = grid(job_name='{recipe}_{short_name}_{seed}u')
    assert g.job_name_axes == ('recipe', 'seed')
    values = {'recipe': 'cost-paper', 'dataset': 'Coffee', 'seed': '7'}
    assert g.job_name_for(study='s', values=values) == 'cost-paper_cls_7u'


def test_job_name_study_placeholder() -> None:
    assert grid(job_name='{study}-{short_name}').job_name_for(study='aug', values={}) == 'aug-cls'


@pytest.mark.parametrize('short_name', ['a.b', 'a b', ''])
def test_short_name_keeps_its_characters(short_name: str) -> None:
    with pytest.raises(ValueError, match='short_name'):
        grid(short_name=short_name)


@pytest.mark.parametrize(
    ('template', 'needle'),
    [
        ('{model}', "unknown placeholder 'model'"),
        ('${HOME}_x', 'not allowed'),
        ('{{recipe}}', 'not allowed'),
        ('a b', 'not allowed'),
        ('a/b', 'not allowed'),
    ],
)
def test_bad_job_name_templates_fail(template: str, needle: str) -> None:
    with pytest.raises(ValueError, match=re.escape(needle)):
        grid(job_name=template)


@pytest.mark.parametrize(
    'axis', ['short_name', 'study', 'output_root', 'log_file', 'cpus', 'cpus_per_run']
)
def test_placeholder_axis_names_are_reserved(axis: str) -> None:
    with pytest.raises(ValueError, match='reserved'):
        grid(axes={axis: {'a': {'seed': 1}}})


def test_job_names_must_differ_across_grids() -> None:
    with pytest.raises(ValueError, match=r"grid 'b'.*job name 'cost-paper'.*grid 'a'"):
        Study(
            name='s',
            grids={
                'a': grid(short_name='a', job_name='{recipe}'),
                'b': grid(short_name='b', job_name='{recipe}'),
            },
        )


def test_template_without_short_name_is_fine_when_unique() -> None:
    Study(
        name='s',
        grids={
            'a': grid(short_name='a', job_name='{recipe}_x'),
            'b': grid(short_name='b', job_name='{recipe}_y'),
        },
    )


def test_job_name_must_not_clash_with_followups() -> None:
    with pytest.raises(ValueError, match="follow-up job name 's_retry'"):
        Study(name='s', grids={'a': grid(short_name='retry')})


def test_study_job_names() -> None:
    study = Study(name='s', grids={'a': grid(short_name='a', job_name='{short_name}_{seed}')})
    assert study_job_names(study=study) == {'a': frozenset({'a_42', 'a_7'})}

"""``Study``, ``Grid`` and ``Override``: shapes and names, pinned at build time."""

import pytest

from tests.lab.unit.study.configs import grid
from tslab.lab.study import Override, setting_axis, Study
from tslab.lab.study.definition.definition import merge_overrides


def test_setting_axis() -> None:
    assert setting_axis(path='model.lr', values=(0.1, 1e-3)) == {
        '0.1': {'model.lr': 0.1},
        '0.001': {'model.lr': 1e-3},
    }


@pytest.mark.parametrize('name', ['', '.', '..', 'a/b', 'a b', 'ü'])
def test_bad_names_raise(name: str) -> None:
    with pytest.raises(ValueError, match='name'):
        Study(name=name, grids={'g': grid()})
    with pytest.raises(ValueError, match='name'):
        Study(name='s', grids={name: grid()})
    with pytest.raises(ValueError, match='name'):
        grid(axes={name: {'a': {}}})
    with pytest.raises(ValueError, match='name'):
        grid(axes={'seed': {name: {'seed': 1}}})


def test_empty_things_raise() -> None:
    with pytest.raises(ValueError, match='grid'):
        Study(name='s', grids={})
    with pytest.raises(ValueError, match='axis'):
        grid(axes={})
    with pytest.raises(ValueError, match='seed'):
        grid(axes={'seed': {}})
    with pytest.raises(ValueError, match='command'):
        grid(command=())


@pytest.mark.parametrize('short_name', ['', 'a.b', 'a b'])
def test_bad_short_names_raise(short_name: str) -> None:
    with pytest.raises(ValueError, match='short_name'):
        grid(short_name=short_name)


def test_short_names_are_unique_in_a_study() -> None:
    with pytest.raises(ValueError, match='cls'):
        Study(name='s', grids={'a': grid(), 'b': grid()})


def test_grid_with_every_run_excluded_raises_even_when_another_grid_has_runs() -> None:
    other = grid(short_name='other')
    empty = grid(exclude=({'dataset': 'Coffee'},))
    with pytest.raises(ValueError, match="grid 'b'"):
        Study(name='s', grids={'a': other, 'b': empty})


@pytest.mark.parametrize(
    'axis', ['grid', 'run_path', 'study_state', 'instance', 'instance_state', 'seconds', 'error']
)
def test_axis_names_that_are_result_columns_are_reserved(axis: str) -> None:
    with pytest.raises(ValueError, match='reserved'):
        grid(axes={axis: {'a': {}}})


@pytest.mark.parametrize('name', ['.study', '.hidden'])
def test_names_starting_with_a_dot_raise(name: str) -> None:
    with pytest.raises(ValueError, match='name'):
        Study(name='s', grids={name: grid()})
    with pytest.raises(ValueError, match='name'):
        Study(name=name, grids={'g': grid()})


def test_exclude_without_trailing_comma_hints_at_it() -> None:
    with pytest.raises(TypeError, match='trailing comma'):
        grid(exclude={'dataset': 'Coffee'})
    with pytest.raises(TypeError, match='exclude'):
        grid(exclude=('dataset=Coffee',))


def test_overrides_without_trailing_comma_hint_at_it() -> None:
    with pytest.raises(TypeError, match='trailing comma'):
        grid(overrides=Override(match={'seed': '42'}, settings={}))
    with pytest.raises(TypeError, match='Override'):
        grid(overrides=({'seed': '42'},))


def test_axis_value_must_be_a_mapping_of_settings() -> None:
    with pytest.raises(TypeError, match=r"axis 'seed' value 'a'"):
        grid(axes={'seed': {'a': ('seed', 1)}})
    with pytest.raises(TypeError, match=r"axis 'seed'"):
        grid(axes={'seed': [1, 2]})


def test_match_values_must_be_names() -> None:
    with pytest.raises(TypeError, match='seed'):
        Study(name='s', grids={'g': grid(exclude=({'seed': 42},))})


def test_setting_axis_rejects_duplicate_values() -> None:
    with pytest.raises(ValueError, match='twice'):
        setting_axis(path='seed', values=(1, 1))


def test_grid_level_match_naming_another_grid_says_so() -> None:
    with pytest.raises(ValueError, match='own grid'):
        Study(name='s', grids={'a': grid(exclude=({'grid': 'b'},)), 'b': grid(short_name='b')})


def test_override_switching_a_choice_drops_children_of_the_old_choice() -> None:
    result = merge_overrides(
        base={'model': 'ts2vec', 'model.depth': 3, 'seed': 1},
        overrides=[({'seed': '1'}, {'model': 'cost'})],
        grid_name='g',
        values={'seed': '1'},
    )
    assert result == {'model': 'cost', 'seed': 1}


def test_override_keeping_the_choice_keeps_its_children() -> None:
    result = merge_overrides(
        base={'model': 'ts2vec', 'model.depth': 3},
        overrides=[({'seed': '1'}, {'model': 'ts2vec', 'model.lr': 0.1})],
        grid_name='g',
        values={'seed': '1'},
    )
    assert result == {'model': 'ts2vec', 'model.depth': 3, 'model.lr': 0.1}

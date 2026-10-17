"""``Study``, ``Grid`` and ``Override``: shapes and names, pinned at build time."""

import pytest

from study.configs import grid
from tslab.study import setting_axis, Study


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


@pytest.mark.parametrize('short_name', ['', 'longer_than8', 'a.b', 'a b'])
def test_bad_short_names_raise(short_name: str) -> None:
    with pytest.raises(ValueError, match='short_name'):
        grid(short_name=short_name)


def test_short_names_are_unique_in_a_study() -> None:
    with pytest.raises(ValueError, match='cls'):
        Study(name='s', grids={'a': grid(), 'b': grid()})

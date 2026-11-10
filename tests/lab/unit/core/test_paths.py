from pathlib import Path

import pytest

from tslab.lab.core.paths import (
    normalize_instance_name,
    require_path,
    validate_relative_path,
    validate_stage_name,
)


def test_require_path_rejects_str() -> None:
    with pytest.raises(TypeError, match=r'must be pathlib\.Path, got str'):
        require_path(value='runs', parameter='root')


def test_require_path_accepts_path(tmp_path: Path) -> None:
    assert require_path(value=tmp_path, parameter='root') is tmp_path


@pytest.mark.parametrize('name', ['..', '.', 'a/b', 'a b', '', 'é'])
def test_invalid_name_parts_are_rejected(name: str) -> None:
    with pytest.raises(ValueError, match='invalid name part'):
        normalize_instance_name(name=name)


def test_instance_name_forms() -> None:
    assert normalize_instance_name(name=None) == ('default',)
    assert normalize_instance_name(name='smooth') == ('smooth',)
    assert normalize_instance_name(name=('forecasting', 'h24', 'smooth')) == (
        'forecasting',
        'h24',
        'smooth',
    )


def test_instance_name_rejects_empty_tuple_and_lists() -> None:
    with pytest.raises(ValueError, match='empty tuple'):
        normalize_instance_name(name=())
    with pytest.raises(TypeError, match='str or tuple'):
        normalize_instance_name(name=['a'])  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize('name', ['logs', 'trackers'])
def test_reserved_stage_names(name: str) -> None:
    with pytest.raises(ValueError, match='reserved'):
        validate_stage_name(name=name)


@pytest.mark.parametrize('path', [Path('/abs/run'), Path(), Path('a/../b')])
def test_relative_path_validation(path: Path) -> None:
    with pytest.raises(ValueError, match=r'relative_path|invalid name part'):
        validate_relative_path(path=path, parameter='relative_path')


def test_relative_path_rejects_str() -> None:
    with pytest.raises(TypeError):
        validate_relative_path(
            path='a/b',  # ty: ignore[invalid-argument-type]
            parameter='relative_path',
        )


@pytest.mark.parametrize('name', ['.hidden', '.study', 'x.tmp-1', 'a.tmp-old-1'])
def test_leading_dot_and_temporary_marker_are_rejected(name: str) -> None:
    with pytest.raises(ValueError, match='invalid name part'):
        normalize_instance_name(name=name)


@pytest.mark.parametrize('name', ['logs', 'trackers', 'status.json', 'meta.json', '.lock'])
def test_names_written_into_a_run_folder_are_reserved(name: str) -> None:
    with pytest.raises(ValueError, match=r'invalid name part|reserved'):
        validate_stage_name(name=name)

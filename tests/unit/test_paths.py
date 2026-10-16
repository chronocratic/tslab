from pathlib import Path

import pytest

from tslab.paths import (
    atomic_write_text,
    normalize_instance_name,
    replace_directory,
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
    assert normalize_instance_name(name='fgsm') == ('fgsm',)
    assert normalize_instance_name(name=('forecasting', 'h24', 'fgsm')) == (
        'forecasting',
        'h24',
        'fgsm',
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
        validate_relative_path(path='a/b', parameter='relative_path')  # ty: ignore[invalid-argument-type]


def test_atomic_write_leaves_old_file_when_rename_never_happens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / 'status.json'
    target.write_text('old', encoding='utf-8')

    def crash(*_args: object, **_kwargs: object) -> None:
        message = 'simulated crash before rename'
        raise OSError(message)

    monkeypatch.setattr(Path, 'replace', crash)
    with pytest.raises(OSError, match='simulated crash'):
        atomic_write_text(path=target, text='new')
    assert target.read_text(encoding='utf-8') == 'old'


def test_atomic_write_creates_parents(tmp_path: Path) -> None:
    target = tmp_path / 'a' / 'b.txt'
    atomic_write_text(path=target, text='x')
    assert target.read_text(encoding='utf-8') == 'x'
    assert [item.name for item in target.parent.iterdir()] == ['b.txt']


def test_replace_directory_replaces_existing(tmp_path: Path) -> None:
    source = tmp_path / 'x.tmp-1'
    source.mkdir()
    (source / 'new.txt').write_text('new', encoding='utf-8')
    destination = tmp_path / 'x'
    destination.mkdir()
    (destination / 'old.txt').write_text('old', encoding='utf-8')
    replace_directory(source=source, destination=destination)
    assert sorted(item.name for item in destination.iterdir()) == ['new.txt']
    assert not source.exists()

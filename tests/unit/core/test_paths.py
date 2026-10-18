from pathlib import Path
import threading

import pytest

from tslab.core.paths import (
    atomic_write_json,
    atomic_write_text,
    CONFIG_FILE,
    ERROR_FILE,
    JsonFileError,
    LOCK_FILE,
    META_FILE,
    normalize_instance_name,
    read_json_object,
    replace_directory,
    require_path,
    STATUS_FILE,
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


@pytest.mark.parametrize('name', ['.hidden', '.study', 'x.tmp-1', 'a.tmp-old-1'])
def test_leading_dot_and_temporary_marker_are_rejected(name: str) -> None:
    with pytest.raises(ValueError, match='invalid name part'):
        normalize_instance_name(name=name)


@pytest.mark.parametrize('name', ['logs', 'trackers', 'status.json', 'meta.json', '.lock'])
def test_names_written_into_a_run_folder_are_reserved(name: str) -> None:
    with pytest.raises(ValueError, match=r'invalid name part|reserved'):
        validate_stage_name(name=name)


def test_on_disk_names_are_public_constants() -> None:
    assert (STATUS_FILE, META_FILE, LOCK_FILE) == ('status.json', 'meta.json', '.lock')
    assert (CONFIG_FILE, ERROR_FILE) == ('config.yaml', 'error.txt')


def test_failed_write_leaves_no_temporary_file(tmp_path: Path) -> None:
    target = tmp_path / 'a.json'
    atomic_write_text(path=target, text='old')
    with pytest.raises(OSError, match='Is a directory'):
        atomic_write_text(path=tmp_path, text='x')
    assert sorted(item.name for item in tmp_path.iterdir()) == ['a.json']


def test_atomic_write_json_is_sorted_and_indented(tmp_path: Path) -> None:
    atomic_write_json(path=tmp_path / 'a.json', document={'b': 1, 'a': [1]})
    assert (tmp_path / 'a.json').read_text(encoding='utf-8') == (
        '{\n  "a": [\n    1\n  ],\n  "b": 1\n}\n'
    )


def test_concurrent_writes_from_threads_never_collide(tmp_path: Path) -> None:
    target = tmp_path / 'a.txt'
    errors: list[BaseException] = []

    def write(value: str) -> None:
        try:
            for _ in range(20):
                atomic_write_text(path=target, text=value)
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=write, args=(str(index) * 50,)) for index in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    assert len(set(target.read_text(encoding='utf-8'))) == 1
    assert [item.name for item in tmp_path.iterdir()] == ['a.txt']


def test_read_json_object_names_the_file_on_bad_content(tmp_path: Path) -> None:
    (tmp_path / 'a.json').write_text('[1]', encoding='utf-8')
    with pytest.raises(JsonFileError, match=r'a\.json must be a JSON object'):
        read_json_object(path=tmp_path / 'a.json')
    (tmp_path / 'b.json').write_text('{broken', encoding='utf-8')
    with pytest.raises(JsonFileError, match=r'b\.json is not valid JSON'):
        read_json_object(path=tmp_path / 'b.json')
    (tmp_path / 'c.json').write_bytes(b'\xff\xfe')
    with pytest.raises(JsonFileError, match=r'c\.json'):
        read_json_object(path=tmp_path / 'c.json')
    with pytest.raises(FileNotFoundError):
        read_json_object(path=tmp_path / 'missing.json')


def test_replace_directory_restores_the_old_folder_when_the_swap_fails(tmp_path: Path) -> None:
    destination = tmp_path / 'final'
    destination.mkdir()
    (destination / 'old.txt').write_text('old', encoding='utf-8')
    with pytest.raises(FileNotFoundError):
        replace_directory(source=tmp_path / 'missing', destination=destination)
    assert (destination / 'old.txt').read_text(encoding='utf-8') == 'old'
    assert [item.name for item in tmp_path.iterdir()] == ['final']

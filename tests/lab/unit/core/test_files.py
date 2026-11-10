from pathlib import Path
import threading

import pytest

from tslab.lab.core.files import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    JsonFileError,
    read_json_object,
    replace_directory,
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


def test_atomic_write_bytes_writes_and_creates_parents(tmp_path: Path) -> None:
    target = tmp_path / 'a' / 'b.png'
    atomic_write_bytes(path=target, data=b'\x89PNG')
    assert target.read_bytes() == b'\x89PNG'
    assert [item.name for item in target.parent.iterdir()] == ['b.png']


def test_atomic_write_bytes_keeps_old_file_on_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / 'b.bin'
    target.write_bytes(b'old')

    def crash(*_args: object, **_kwargs: object) -> None:
        message = 'simulated crash before rename'
        raise OSError(message)

    monkeypatch.setattr(Path, 'replace', crash)
    with pytest.raises(OSError, match='simulated crash'):
        atomic_write_bytes(path=target, data=b'new')
    assert target.read_bytes() == b'old'
    assert [item.name for item in tmp_path.iterdir()] == ['b.bin']

import json
from pathlib import Path

import pytest

from tslab.lab.core.artifact import Checkpoint

BEST = 'epoch=3-val_loss_epoch=0.10.ckpt'


def write_files(*, directory: Path, names: tuple[str, ...]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_bytes(name.encode())


def test_save_writes_only_the_names(tmp_path: Path) -> None:
    write_files(directory=tmp_path, names=(BEST, 'last.ckpt'))
    Checkpoint(directory=tmp_path, best=BEST, last='last.ckpt').save(directory=tmp_path)
    document = json.loads((tmp_path / 'checkpoint.json').read_text(encoding='utf-8'))
    assert document == {'best': BEST, 'format_version': 2, 'last': 'last.ckpt', 'trusted': True}
    loaded = Checkpoint.load(directory=tmp_path)
    assert loaded.best_path == tmp_path / BEST
    assert loaded.last_path == tmp_path / 'last.ckpt'
    assert loaded.trusted is True


def test_save_into_another_folder_copies_the_files(tmp_path: Path) -> None:
    source = tmp_path / 'source'
    target = tmp_path / 'target'
    write_files(directory=source, names=(BEST, 'last.ckpt'))
    target.mkdir()
    checkpoint = Checkpoint(directory=source, best=BEST, last='last.ckpt')
    checkpoint.save(directory=target)
    assert checkpoint.best_path == target / BEST
    assert (target / BEST).read_bytes() == BEST.encode()
    assert Checkpoint.load(directory=target).last_path == target / 'last.ckpt'


def test_load_requires_both_files(tmp_path: Path) -> None:
    write_files(directory=tmp_path, names=(BEST, 'last.ckpt'))
    Checkpoint(directory=tmp_path, best=BEST, last='last.ckpt').save(directory=tmp_path)
    (tmp_path / 'last.ckpt').unlink()
    with pytest.raises(FileNotFoundError, match=r'last\.ckpt'):
        Checkpoint.load(directory=tmp_path)


def test_load_rejects_an_unknown_format_version(tmp_path: Path) -> None:
    document = {'best': BEST, 'format_version': 99, 'last': 'last.ckpt', 'trusted': True}
    (tmp_path / 'checkpoint.json').write_text(json.dumps(document), encoding='utf-8')
    with pytest.raises(ValueError, match='format_version'):
        Checkpoint.load(directory=tmp_path)


@pytest.mark.parametrize('name', ['', '.', '..', '../escape.ckpt', 'sub/last.ckpt'])
def test_names_must_be_plain_file_names(tmp_path: Path, *, name: str) -> None:
    with pytest.raises(ValueError, match='plain file name'):
        Checkpoint(directory=tmp_path, best=name, last='last.ckpt')


def test_external_file_is_untrusted_and_fingerprinted_by_content(tmp_path: Path) -> None:
    path = tmp_path / 'pretrained.ckpt'
    path.write_bytes(b'weights v1')
    checkpoint = Checkpoint.external(path=path)
    assert checkpoint.best_path == path
    assert checkpoint.last_path == path
    assert checkpoint.trusted is False
    trusted = Checkpoint.external(path=path, trusted=True)
    assert trusted.trusted is True
    # Trust never changes the fingerprint; content does, the path does not.
    assert trusted.tslab_fingerprint == checkpoint.tslab_fingerprint
    copy = tmp_path / 'copy.ckpt'
    copy.write_bytes(b'weights v1')
    assert Checkpoint.external(path=copy).tslab_fingerprint == checkpoint.tslab_fingerprint
    path.write_bytes(b'weights v2')
    assert Checkpoint.external(path=path).tslab_fingerprint != checkpoint.tslab_fingerprint


def test_external_folder_uses_its_checkpoint_json(tmp_path: Path) -> None:
    write_files(directory=tmp_path, names=(BEST, 'last.ckpt'))
    Checkpoint(directory=tmp_path, best=BEST, last='last.ckpt').save(directory=tmp_path)
    checkpoint = Checkpoint.external(path=tmp_path)
    assert checkpoint.best_path == tmp_path / BEST
    assert checkpoint.trusted is False


def test_external_rejects_missing_files_and_str_paths(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Checkpoint.external(path=tmp_path / 'missing.ckpt')
    with pytest.raises(TypeError, match=r'pathlib\.Path'):
        Checkpoint.external(path='model.ckpt')  # ty: ignore[invalid-argument-type]


def test_untrusted_flag_survives_save_and_load(tmp_path: Path) -> None:
    write_files(directory=tmp_path, names=(BEST, 'last.ckpt'))
    untrusted = Checkpoint(directory=tmp_path, best=BEST, last='last.ckpt', trusted=False)
    untrusted.save(directory=tmp_path)
    assert Checkpoint.load(directory=tmp_path).trusted is False


def test_load_rejects_a_file_without_the_trust_flag(tmp_path: Path) -> None:
    write_files(directory=tmp_path, names=(BEST, 'last.ckpt'))
    document = {'best': BEST, 'format_version': 2, 'last': 'last.ckpt'}
    (tmp_path / 'checkpoint.json').write_text(json.dumps(document), encoding='utf-8')
    with pytest.raises(ValueError, match='trusted'):
        Checkpoint.load(directory=tmp_path)

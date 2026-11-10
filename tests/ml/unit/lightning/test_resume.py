import logging
import os
from pathlib import Path

import pytest
import torch

from tslab.lightning.resume import choose_resume_checkpoint, resume_candidates

MONITOR = 'val_loss_epoch'


def write_checkpoint(*, path: Path, modified: float, valid: bool = True) -> None:
    if valid:
        torch.save(obj={'state_dict': {}, 'loops': {}}, f=path)
    else:
        path.write_bytes(b'truncated by a full disk')
    os.utime(path, (modified, modified))


def make_folder(*, directory: Path, valid_last: bool = True) -> None:
    write_checkpoint(path=directory / 'last.ckpt', modified=500, valid=valid_last)
    write_checkpoint(path=directory / 'epoch=1-val_loss_epoch=0.50.ckpt', modified=100)
    write_checkpoint(path=directory / 'epoch=2-val_loss_epoch=0.20.ckpt', modified=200)
    write_checkpoint(path=directory / 'manual.ckpt', modified=300)


def test_candidate_order(tmp_path: Path) -> None:
    make_folder(directory=tmp_path)
    write_checkpoint(path=tmp_path / 'last-v1.ckpt', modified=600)
    names = [path.name for path in resume_candidates(directory=tmp_path, monitor=MONITOR)]
    assert names == [
        'last-v1.ckpt',
        'last.ckpt',
        'epoch=2-val_loss_epoch=0.20.ckpt',
        'epoch=1-val_loss_epoch=0.50.ckpt',
        'manual.ckpt',
    ]


def test_a_monitor_with_a_slash_is_ranked_by_its_file_name_form(tmp_path: Path) -> None:
    write_checkpoint(path=tmp_path / 'epoch=1-val_loss=0.50.ckpt', modified=100)
    write_checkpoint(path=tmp_path / 'epoch=2-val_loss=0.20.ckpt', modified=50)
    names = [path.name for path in resume_candidates(directory=tmp_path, monitor='val/loss')]
    assert names == ['epoch=2-val_loss=0.20.ckpt', 'epoch=1-val_loss=0.50.ckpt']


def test_last_is_chosen_when_it_loads(tmp_path: Path) -> None:
    make_folder(directory=tmp_path)
    assert choose_resume_checkpoint(directory=tmp_path, monitor=MONITOR) == tmp_path / 'last.ckpt'


def test_corrupt_last_falls_back_to_the_best_checkpoint(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    make_folder(directory=tmp_path, valid_last=False)
    with caplog.at_level(logging.WARNING):
        chosen = choose_resume_checkpoint(directory=tmp_path, monitor=MONITOR)
    assert chosen == tmp_path / 'epoch=2-val_loss_epoch=0.20.ckpt'
    assert 'cannot resume from last.ckpt' in caplog.text


def test_a_dict_without_training_state_is_rejected(tmp_path: Path) -> None:
    torch.save(obj={'state_dict': {}}, f=tmp_path / 'last.ckpt')
    write_checkpoint(path=tmp_path / 'manual.ckpt', modified=100)
    chosen = choose_resume_checkpoint(directory=tmp_path, monitor=MONITOR)
    assert chosen == tmp_path / 'manual.ckpt'


def test_nothing_usable_means_training_from_scratch(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    write_checkpoint(path=tmp_path / 'last.ckpt', modified=100, valid=False)
    with caplog.at_level(logging.WARNING):
        assert choose_resume_checkpoint(directory=tmp_path, monitor=MONITOR) is None
    assert 'training from scratch' in caplog.text


def test_empty_folder_is_a_quiet_fresh_start(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        assert choose_resume_checkpoint(directory=tmp_path, monitor=MONITOR) is None
    assert caplog.text == ''

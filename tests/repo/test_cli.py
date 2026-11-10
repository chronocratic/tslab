from pathlib import Path

import pytest

from tslab.cli import main


def test_default_template_is_ts(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(['new', str(tmp_path / 'p')])
    assert exit_info.value.code == 0
    assert (tmp_path / 'p' / 'src' / 'p' / 'forecast.py').exists()


def test_lab_template_has_no_ts_code(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(['new', str(tmp_path / 'p'), '--template', 'lab'])
    assert exit_info.value.code == 0
    assert (tmp_path / 'p' / 'src' / 'p' / 'stages.py').exists()
    assert not (tmp_path / 'p' / 'src' / 'p' / 'forecast.py').exists()


def test_wandb_sync_is_registered() -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(['wandb-sync', '--help'])
    assert exit_info.value.code == 0

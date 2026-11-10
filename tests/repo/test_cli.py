from collections.abc import Callable
import importlib.util
from pathlib import Path
import shutil

import pytest

from tslab.cli import main
from tslab.lab.cli import main as lab_main


def test_default_template_is_lab_without_torch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    present = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        'find_spec',
        lambda name, *args: None if name == 'torch' else present(name, *args),
    )
    with pytest.raises(SystemExit) as exit_info:
        main(['new', str(tmp_path / 'p')])
    assert exit_info.value.code == 0
    assert not (tmp_path / 'p' / 'src' / 'p' / 'forecast.py').exists()


def test_default_template_is_ts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    present = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        'find_spec',
        lambda name, *args: object() if name == 'torch' else present(name, *args),
    )
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


def _exit_code(*, main_function: Callable[[list[str]], None], argv: list[str]) -> object:
    with pytest.raises(SystemExit) as exit_info:
        main_function(argv)
    return exit_info.value.code


def test_each_entry_point_keeps_its_own_commands_and_templates(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for order in (('full', 'lab'), ('lab', 'full'), ('full', 'lab')):
        for kind in order:
            entry = main if kind == 'full' else lab_main
            _exit_code(main_function=entry, argv=['--help'])
            listing = capsys.readouterr().out
            assert ('wandb-sync' in listing) is (kind == 'full')
            code = _exit_code(
                main_function=entry, argv=['new', str(tmp_path / kind), '--template', 'ts']
            )
            assert code == (0 if kind == 'full' else 1)
            shutil.rmtree(tmp_path / kind, ignore_errors=True)
    assert _exit_code(main_function=lab_main, argv=['new', str(tmp_path / 'plain')]) == 0
    assert not (tmp_path / 'plain' / 'src' / 'plain' / 'forecast.py').exists()

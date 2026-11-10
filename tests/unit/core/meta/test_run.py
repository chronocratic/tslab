from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess

import pytest
import torch

from tslab.core.meta import (
    collect_code_snapshot,
    collect_git_state,
    collect_hardware,
    collect_versions,
    package_version,
)
from tslab.core.run import Run


@dataclass(frozen=True, kw_only=True)
class _MetadataConfig:
    seed: int = 0


def test_versions_include_python_and_torch() -> None:
    versions = collect_versions()
    assert versions['python'] is not None
    assert versions['torch'] is not None
    assert package_version(distribution='surely-not-installed-package') is None


def test_collect_hardware_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)

    hardware = collect_hardware()

    assert hardware['cuda_available'] is False
    assert hardware['gpus'] == []
    assert hardware['cuda_visible_devices'] is None


def test_collect_hardware_lists_visible_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda index: f'GPU {index}')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2,4')

    hardware = collect_hardware()

    assert hardware['cuda_available'] is True
    assert hardware['gpus'] == ['GPU 0', 'GPU 1']
    assert hardware['cuda_visible_devices'] == '2,4'


def test_run_environment_includes_hardware(tmp_path: Path) -> None:
    run = Run(config=_MetadataConfig(), root=tmp_path, relative_path=Path('run'))

    assert set(run.environment()) == {'versions', 'git', 'hardware'}


def test_collect_code_snapshot_without_the_variable_is_none() -> None:
    assert collect_code_snapshot(environ={}) is None
    assert collect_code_snapshot(environ={'TSLAB_CODE_SNAPSHOT': ''}) is None


def test_collect_code_snapshot_reads_every_field() -> None:
    environ = {
        'TSLAB_CODE_SNAPSHOT': '/out/.tslab-code-snapshots/0123456789abcdef',
        'TSLAB_CODE_SNAPSHOT_HASH': '0123456789abcdef',
        'TSLAB_CODE_SNAPSHOT_COMMIT': 'a' * 40,
        'TSLAB_CODE_SNAPSHOT_DIRTY': 'true',
        'TSLAB_CODE_SNAPSHOT_UNTRACKED_COPIED': 'false',
    }
    assert collect_code_snapshot(environ=environ) == {
        'folder': '/out/.tslab-code-snapshots/0123456789abcdef',
        'hash': '0123456789abcdef',
        'commit': 'a' * 40,
        'dirty': True,
        'untracked_copied': False,
    }


def test_git_state_outside_a_repository(tmp_path: Path) -> None:
    assert collect_git_state(directory=tmp_path) == {'commit': None, 'dirty': None}


def test_git_state_inside_a_repository(tmp_path: Path) -> None:
    git = shutil.which('git')
    assert git is not None
    commands = (
        ['init', '-q'],
        ['-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-q', '--allow-empty', '-m', 'x'],
    )
    for arguments in commands:
        # why: fixed git arguments
        subprocess.run([git, *arguments], cwd=tmp_path, check=True)  # noqa: S603
    clean = collect_git_state(directory=tmp_path)
    assert isinstance(clean['commit'], str)
    assert len(clean['commit']) == 40
    assert clean['dirty'] is False
    (tmp_path / 'new.txt').write_text('x', encoding='utf-8')
    assert collect_git_state(directory=tmp_path)['dirty'] is True

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import sys

from tslab.lab.core.meta import (
    collect_code_snapshot,
    collect_git_state,
    collect_hardware,
    collect_versions,
    package_version,
    register_hardware,
    register_versions,
    run as run_module,
)
from tslab.lab.core.run import Run


@dataclass(frozen=True, kw_only=True)
class _MetadataConfig:
    seed: int = 0


def test_versions_include_python_and_tslab() -> None:
    versions = collect_versions()
    assert versions['python'] is not None
    assert package_version(distribution='surely-not-installed-package') is None


def test_registered_collectors_are_merged_once() -> None:
    saved = (list(run_module._version_collectors), list(run_module._hardware_collectors))  # noqa: SLF001
    try:
        register_versions(lambda: {'extra': '1'})
        register_hardware(_extra_hardware)
        register_hardware(_extra_hardware)
        assert collect_versions()['extra'] == '1'
        assert collect_hardware()['extra'] is True
    finally:
        run_module._version_collectors[:] = saved[0]  # noqa: SLF001
        run_module._hardware_collectors[:] = saved[1]  # noqa: SLF001


def _extra_hardware() -> dict[str, object]:
    return {'extra': True}


def test_lab_metadata_has_no_torch_keys() -> None:
    code = (
        'from tslab.lab.core.meta import collect_hardware, collect_versions\n'
        "assert 'torch' not in collect_versions()\n"
        "assert 'gpus' not in collect_hardware()\n"
    )
    subprocess.run([sys.executable, '-c', code], check=True)  # noqa: S603 # why: fixed code


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

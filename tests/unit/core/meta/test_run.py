from pathlib import Path
import shutil
import subprocess

from tslab.core.meta import (
    collect_code_snapshot,
    collect_git_state,
    collect_versions,
    package_version,
)


def test_versions_include_python_and_torch() -> None:
    versions = collect_versions()
    assert versions['python'] is not None
    assert versions['torch'] is not None
    assert package_version(distribution='surely-not-installed-package') is None


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

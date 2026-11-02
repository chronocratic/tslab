"""Code snapshots: the copy of ``src/`` that submitted jobs run from."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest

import tslab.study.code_snapshot as code_snapshot_module
from tslab.study.code_snapshot import (
    CODE_SNAPSHOTS_FOLDER,
    CodeSnapshot,
    CodeSnapshotError,
    take_code_snapshot,
    without_code_snapshot,
)


def _git(*, cwd: Path, arguments: tuple[str, ...]) -> None:
    git = shutil.which('git')
    assert git is not None
    subprocess.run(  # noqa: S603 # why: fixed git arguments
        [git, '-c', 'user.name=t', '-c', 'user.email=t@t', *arguments],
        cwd=cwd,
        capture_output=True,
        check=True,
    )


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / 'project'
    (root / 'src' / 'pkg').mkdir(parents=True)
    (root / 'README.md').write_text('# project\n', encoding='utf-8')
    (root / 'src' / 'pkg' / '__init__.py').write_text('X = 1\n', encoding='utf-8')
    (root / 'src' / 'pkg' / 'model.py').write_text('Y = 1\n', encoding='utf-8')
    _git(cwd=root, arguments=('init', '-q'))
    _git(cwd=root, arguments=('add', '-A'))
    _git(cwd=root, arguments=('commit', '-q', '-m', 'init'))
    return root


def _take(
    *,
    project_root: Path,
    output_root: Path,
    include_untracked: bool = False,
    check_untracked: bool = True,
    dry_run: bool = False,
) -> CodeSnapshot:
    return take_code_snapshot(
        project_root=project_root,
        output_root=output_root,
        include_untracked=include_untracked,
        check_untracked=check_untracked,
        dry_run=dry_run,
    )


def test_copies_tracked_src_with_working_tree_bytes(project: Path, tmp_path: Path) -> None:
    (project / 'src' / 'pkg' / '__init__.py').write_text('X = 2\n', encoding='utf-8')
    taken = _take(project_root=project, output_root=tmp_path / 'out')
    assert (taken.folder / 'src' / 'pkg' / '__init__.py').read_text(encoding='utf-8') == 'X = 2\n'
    assert taken.dirty is True
    assert taken.folder.parent == tmp_path / 'out' / CODE_SNAPSHOTS_FOLDER
    assert len(taken.hash) == 16
    assert taken.folder.name == taken.hash
    assert not (taken.folder / 'README.md').exists()  # only src/


def test_same_code_same_folder_reused(project: Path, tmp_path: Path) -> None:
    out = tmp_path / 'out'
    first = _take(project_root=project, output_root=out)
    marker = first.folder / 'src' / 'extra-note.txt'
    marker.write_text('kept\n', encoding='utf-8')
    second = _take(project_root=project, output_root=out)
    assert second.folder == first.folder
    assert marker.read_text(encoding='utf-8') == 'kept\n'


def test_reused_folder_that_lost_src_is_recopied(project: Path, tmp_path: Path) -> None:
    out = tmp_path / 'out'
    first = _take(project_root=project, output_root=out)
    shutil.rmtree(first.folder / 'src')
    second = _take(project_root=project, output_root=out)
    assert second.folder == first.folder
    assert (second.folder / 'src' / 'pkg' / '__init__.py').is_file()
    assert (second.folder / 'src' / 'pkg' / 'model.py').is_file()


def test_changed_code_new_folder_no_stale_files(project: Path, tmp_path: Path) -> None:
    first = _take(project_root=project, output_root=tmp_path / 'out')
    (project / 'src' / 'pkg' / 'model.py').unlink()
    _git(cwd=project, arguments=('commit', '-q', '-a', '-m', 'drop model'))
    second = _take(project_root=project, output_root=tmp_path / 'out')
    assert second.folder != first.folder
    assert not (second.folder / 'src' / 'pkg' / 'model.py').exists()


def test_deleted_tracked_file_is_skipped(project: Path, tmp_path: Path) -> None:
    (project / 'src' / 'pkg' / 'model.py').unlink()
    taken = _take(project_root=project, output_root=tmp_path / 'out')
    assert not (taken.folder / 'src' / 'pkg' / 'model.py').exists()


def test_untracked_stops_with_listed_files(project: Path, tmp_path: Path) -> None:
    (project / 'src' / 'pkg' / 'new.py').write_text('Z = 1\n', encoding='utf-8')
    with pytest.raises(CodeSnapshotError) as raised:
        _take(project_root=project, output_root=tmp_path / 'out')
    message = str(raised.value)
    assert message.startswith('1 files under src/ are not tracked by git')
    assert '  src/pkg/new.py' in message
    assert '--code-snapshot-include-untracked' in message
    assert not (tmp_path / 'out').exists()


def test_include_untracked_copies_and_records_it(project: Path, tmp_path: Path) -> None:
    (project / 'src' / 'pkg' / 'new.py').write_text('Z = 1\n', encoding='utf-8')
    taken = _take(project_root=project, output_root=tmp_path / 'out', include_untracked=True)
    assert taken.untracked_copied is True
    assert (taken.folder / 'src' / 'pkg' / 'new.py').is_file()


def test_check_untracked_off_copies_without_stop(project: Path, tmp_path: Path) -> None:
    (project / 'src' / 'pkg' / 'new.py').write_text('Z = 1\n', encoding='utf-8')
    taken = _take(project_root=project, output_root=tmp_path / 'out', check_untracked=False)
    assert taken.untracked_copied is True
    assert (taken.folder / 'src' / 'pkg' / 'new.py').is_file()


def test_ignored_files_never_copied(project: Path, tmp_path: Path) -> None:
    (project / '.gitignore').write_text('*.log\n', encoding='utf-8')
    (project / 'src' / 'pkg' / 'x.log').write_text('log\n', encoding='utf-8')
    taken = _take(project_root=project, output_root=tmp_path / 'out', include_untracked=True)
    assert not (taken.folder / 'src' / 'pkg' / 'x.log').exists()


def test_dry_run_writes_nothing(project: Path, tmp_path: Path) -> None:
    out = tmp_path / 'out'
    dry = _take(project_root=project, output_root=out, dry_run=True)
    assert dry.folder.parent == out / CODE_SNAPSHOTS_FOLDER
    assert not out.exists()
    real = _take(project_root=project, output_root=out)
    assert real.hash == dry.hash
    assert (real.folder / 'src' / 'pkg' / '__init__.py').is_file()


def test_code_changed_while_copying_stops_without_folder(
    project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / 'out'
    real_copy = code_snapshot_module._copy_files  # noqa: SLF001 # why: test the copy hook

    def _copy_and_tamper(*, project_root: Path, copies: list[str], destination: Path) -> None:
        real_copy(project_root=project_root, copies=copies, destination=destination)
        (destination / copies[0]).write_text('tampered\n', encoding='utf-8')

    monkeypatch.setattr(code_snapshot_module, '_copy_files', _copy_and_tamper)
    with pytest.raises(CodeSnapshotError, match='changed while the code snapshot was being copied'):
        _take(project_root=project, output_root=out)
    assert list((out / CODE_SNAPSHOTS_FOLDER).iterdir()) == []


def test_copy_error_is_a_code_snapshot_error(
    project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _broken_copy(*, project_root: Path, copies: list[str], destination: Path) -> None:
        message = 'enospc'
        raise OSError(message)

    monkeypatch.setattr(code_snapshot_module, '_copy_files', _broken_copy)
    out = tmp_path / 'out'
    with pytest.raises(CodeSnapshotError) as raised:
        _take(project_root=project, output_root=out)
    assert 'cannot copy the code snapshot to' in str(raised.value)
    assert list((out / CODE_SNAPSHOTS_FOLDER).iterdir()) == []


def test_not_a_git_repository_is_an_error_naming_no_code_snapshot(tmp_path: Path) -> None:
    with pytest.raises(CodeSnapshotError) as raised:
        _take(project_root=tmp_path, output_root=tmp_path / 'out')
    assert '--no-code-snapshot' in str(raised.value)


def test_no_commit_is_an_error(tmp_path: Path) -> None:
    (tmp_path / 'src').mkdir()
    _git(cwd=tmp_path, arguments=('init', '-q'))
    with pytest.raises(CodeSnapshotError) as raised:
        _take(project_root=tmp_path, output_root=tmp_path / 'out')
    assert '--no-code-snapshot' in str(raised.value)


def test_no_src_folder_is_an_error_naming_src_layout(tmp_path: Path) -> None:
    _git(cwd=tmp_path, arguments=('init', '-q'))
    _git(cwd=tmp_path, arguments=('commit', '-q', '--allow-empty', '-m', 'init'))
    with pytest.raises(CodeSnapshotError) as raised:
        _take(project_root=tmp_path, output_root=tmp_path / 'out')
    assert 'src' in str(raised.value)
    assert '--no-code-snapshot' in str(raised.value)


def test_without_code_snapshot_drops_variables_and_pythonpath_entry() -> None:
    environ = {
        'TSLAB_CODE_SNAPSHOT': '/s',
        'TSLAB_CODE_SNAPSHOT_HASH': 'h',
        'PYTHONPATH': '/s/src:/other',
        'HOME': '/h',
    }
    assert without_code_snapshot(environ=environ) == {'PYTHONPATH': '/other', 'HOME': '/h'}
    assert (
        without_code_snapshot(environ={'TSLAB_CODE_SNAPSHOT': '/s', 'PYTHONPATH': '/s/src'}) == {}
    )


def test_code_snapshot_json_round_trip(tmp_path: Path) -> None:
    taken = CodeSnapshot(
        folder=tmp_path / '.tslab-code-snapshots' / 'abc123',
        hash='abc123',
        commit='a' * 40,
        dirty=True,
        untracked_copied=False,
    )
    document = taken.to_json()
    assert document == {
        'folder': (tmp_path / '.tslab-code-snapshots' / 'abc123').as_posix(),
        'hash': 'abc123',
        'commit': 'a' * 40,
        'dirty': True,
        'untracked_copied': False,
    }
    assert CodeSnapshot.from_json(document=document) == taken


def test_pythonpath_prepends(tmp_path: Path) -> None:
    taken = CodeSnapshot(
        folder=tmp_path / 's', hash='abc123', commit='a' * 40, dirty=False, untracked_copied=False
    )
    assert taken.pythonpath(current=None) == (tmp_path / 's' / 'src').as_posix()
    assert taken.pythonpath(current='/a') == f'{(tmp_path / "s" / "src").as_posix()}:/a'


def test_code_snapshot_variables() -> None:
    taken = CodeSnapshot(
        folder=Path('/s'), hash='abc123', commit='a' * 40, dirty=True, untracked_copied=True
    )
    assert taken.variables() == {
        'TSLAB_CODE_SNAPSHOT': '/s',
        'TSLAB_CODE_SNAPSHOT_HASH': 'abc123',
        'TSLAB_CODE_SNAPSHOT_COMMIT': 'a' * 40,
        'TSLAB_CODE_SNAPSHOT_DIRTY': 'true',
        'TSLAB_CODE_SNAPSHOT_UNTRACKED_COPIED': 'true',
    }


@pytest.mark.slow
def test_uv_run_imports_the_code_snapshot(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    taken = take_code_snapshot(
        project_root=repo_root,
        output_root=tmp_path / 'out',
        include_untracked=True,
        check_untracked=True,
        dry_run=False,
    )
    result = subprocess.run(  # why: fixed command of this repository
        ['uv', 'run', 'python', '-c', 'import tslab; print(tslab.__file__)'],  # noqa: S607
        cwd=repo_root,
        env={**os.environ, **taken.variables(), 'PYTHONPATH': taken.pythonpath(current=None)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().startswith(taken.folder.as_posix())

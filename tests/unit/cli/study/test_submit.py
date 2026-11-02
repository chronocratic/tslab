"""The ``tslab study submit`` command: the messages of its run."""

import logging
from pathlib import Path
import shutil
import subprocess
from typing import cast, TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Sequence

from tests.support.fake_backend import FakeBackend
from tests.unit.cli.study.helpers import toy_project
from tests.unit.study.helpers import jobs_of
from tslab.cli import main
from tslab.cli.study import Submit


def test_dry_run_says_nothing_was_submitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    toy_project(tmp_path=tmp_path, monkeypatch=monkeypatch)

    with caplog.at_level(logging.INFO):
        code = Submit(study='toy_studies:study', machine='local.toml', dry_run=True).run()

    assert code == 0
    assert 'dry run: nothing submitted' in caplog.text


def run_main(*args: str) -> int:
    with pytest.raises(SystemExit) as exit_info:
        main(args=['study', *args])
    code = exit_info.value.code
    assert isinstance(code, int)
    return code


def _git_project(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code_snapshot: bool = False
) -> Path:
    """The toy project as a committed git repository with a src layout and tslab.toml."""
    project = toy_project(tmp_path=tmp_path, monkeypatch=monkeypatch)
    machines = project / 'machines'
    machines.mkdir()
    (machines / 'local.toml').write_text((project / 'local.toml').read_text(encoding='utf-8'))
    (project / 'src' / 'pkg').mkdir(parents=True)
    (project / 'src' / 'pkg' / '__init__.py').write_text('# package\n', encoding='utf-8')
    toml = 'studies = "toy_studies"\n'
    if code_snapshot:
        toml += 'code_snapshot = true\n'
    toml += '\n[machine_by_host]\n"*" = "local"\n'
    (project / 'tslab.toml').write_text(toml, encoding='utf-8')
    git = shutil.which('git')
    assert git is not None
    for arguments in (('init', '-q'), ('add', '-A'), ('commit', '-q', '-m', 'init')):
        subprocess.run(  # noqa: S603 # why: fixed git command, no user input
            [git, '-c', 'user.name=t', '-c', 'user.email=t@t', *arguments],
            cwd=project,
            check=True,
            capture_output=True,
        )
    return project


def test_submit_code_snapshot_records_and_copies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_backend: FakeBackend
) -> None:
    project = _git_project(tmp_path=tmp_path, monkeypatch=monkeypatch)

    assert run_main('submit', 'study', '--machine', 'local', '--code-snapshot') == 0

    snapshot = jobs_of(project=project)[-1].code_snapshot
    assert snapshot is not None
    assert snapshot.folder.parent == project / 'runs' / '.tslab-code-snapshots'
    assert snapshot.folder.is_dir()
    assert (snapshot.folder / 'src' / 'pkg' / '__init__.py').is_file()


def test_code_snapshot_in_toml_turns_it_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_backend: FakeBackend
) -> None:
    project = _git_project(tmp_path=tmp_path, monkeypatch=monkeypatch, code_snapshot=True)

    assert run_main('submit', 'study', '--machine', 'local') == 0

    assert jobs_of(project=project)[-1].code_snapshot is not None


def test_no_code_snapshot_overrides_toml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_backend: FakeBackend
) -> None:
    project = _git_project(tmp_path=tmp_path, monkeypatch=monkeypatch, code_snapshot=True)

    assert run_main('submit', 'study', '--machine', 'local', '--no-code-snapshot') == 0

    assert jobs_of(project=project)[-1].code_snapshot is None


def test_code_snapshot_and_no_code_snapshot_is_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_backend: FakeBackend
) -> None:
    _git_project(tmp_path=tmp_path, monkeypatch=monkeypatch)

    assert (
        run_main('submit', 'study', '--machine', 'local', '--code-snapshot', '--no-code-snapshot')
        == 2
    )


def test_untracked_stop_exit_2_writes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_backend: FakeBackend,
    caplog: pytest.LogCaptureFixture,
) -> None:
    project = _git_project(tmp_path=tmp_path, monkeypatch=monkeypatch)
    (project / 'src' / 'pkg' / 'new.py').write_text('NEW = 1\n', encoding='utf-8')
    with caplog.at_level(logging.ERROR):
        code = run_main('submit', 'study', '--machine', 'local', '--code-snapshot')

    assert code == 2
    assert '--code-snapshot-include-untracked' in caplog.text
    assert not (project / 'runs' / 's' / '.study' / 'submissions.jsonl').exists()
    assert not (project / 'runs' / '.tslab-code-snapshots').exists()


def test_dry_run_code_snapshot_copies_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_backend: FakeBackend
) -> None:
    project = _git_project(tmp_path=tmp_path, monkeypatch=monkeypatch)

    assert run_main('submit', 'study', '--machine', 'local', '--dry-run', '--code-snapshot') == 0

    assert not (project / 'runs' / '.tslab-code-snapshots').exists()


def test_submit_from_subfolder_copies_project_src(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_backend: FakeBackend
) -> None:
    project = _git_project(tmp_path=tmp_path, monkeypatch=monkeypatch)
    # why: the toy study module is not installed; the real project's package is
    monkeypatch.syspath_prepend(str(project))
    monkeypatch.chdir(project / 'src' / 'pkg')

    assert run_main('submit', 'study', '--machine', 'local', '--code-snapshot') == 0

    snapshot = jobs_of(project=project)[-1].code_snapshot
    assert snapshot is not None
    assert snapshot.folder.is_dir()
    assert (snapshot.folder / 'src' / 'pkg' / '__init__.py').is_file()
    assert not (snapshot.folder / 'toy_script.py').exists()


def test_include_untracked_without_code_snapshot_is_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_backend: FakeBackend
) -> None:
    _git_project(tmp_path=tmp_path, monkeypatch=monkeypatch)

    assert (
        run_main('submit', 'study', '--machine', 'local', '--code-snapshot-include-untracked') == 2
    )


def test_detach_forwards_code_snapshot_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_backend: FakeBackend
) -> None:
    _git_project(tmp_path=tmp_path, monkeypatch=monkeypatch)

    code = Submit(
        study='toy_studies:study',
        machine='local',
        detach=True,
        code_snapshot=True,
        code_snapshot_include_untracked=True,
    ).run()

    assert code == 0
    arguments = cast('Sequence[str]', fake_backend.calls('detach')[0]['arguments'])
    assert '--code-snapshot' in arguments
    assert '--code-snapshot-include-untracked' in arguments
    assert '--no-code-snapshot' not in arguments

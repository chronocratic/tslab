"""The ``tslab study resubmit`` command: the code snapshot of its runs."""

import logging
from pathlib import Path
import shutil

import pytest

from tests.support.fake_backend import FakeBackend
from tests.unit.cli.study.helpers import git_project
from tests.unit.study.helpers import jobs_of
from tslab.cli import main
from tslab.study.launchers import JobPhase


def run_main(*args: str) -> int:
    with pytest.raises(SystemExit) as exit_info:
        main(args=['study', *args])
    code = exit_info.value.code
    assert isinstance(code, int)
    return code


def _fail_all(*, project: Path, fake_backend: FakeBackend) -> None:
    """Every recorded task ends failed, so the whole study is resubmittable."""
    phases = {
        f'{job.job_id}_{task.index}': JobPhase.FAILED
        for job in jobs_of(project=project)
        for task in job.tasks
    }
    fake_backend.set_states(phases=phases)


def test_resubmit_reuses_recorded_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_backend: FakeBackend
) -> None:
    project = git_project(tmp_path=tmp_path, monkeypatch=monkeypatch)
    assert run_main('submit', 'study', '--machine', 'local', '--code-snapshot') == 0
    recorded = jobs_of(project=project)[-1].code_snapshot
    assert recorded is not None
    _fail_all(project=project, fake_backend=fake_backend)
    assert run_main('resubmit', 'study', '--machine', 'local', '--states', 'failed') == 0
    latest = jobs_of(project=project)[-1].code_snapshot
    assert latest is not None
    assert latest.folder == recorded.folder
    assert (latest.folder / 'src' / 'pkg' / '__init__.py').is_file()


def test_new_code_snapshot_takes_a_fresh_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_backend: FakeBackend
) -> None:
    project = git_project(tmp_path=tmp_path, monkeypatch=monkeypatch)
    assert run_main('submit', 'study', '--machine', 'local', '--code-snapshot') == 0
    recorded = jobs_of(project=project)[-1].code_snapshot
    assert recorded is not None
    _fail_all(project=project, fake_backend=fake_backend)
    # why: the fresh snapshot differs by the working-tree bytes of one file
    (project / 'src' / 'pkg' / '__init__.py').write_text('# package v2\n', encoding='utf-8')
    assert (
        run_main(
            'resubmit', 'study', '--machine', 'local', '--states', 'failed', '--new-code-snapshot'
        )
        == 0
    )
    latest = jobs_of(project=project)[-1].code_snapshot
    assert latest is not None
    assert latest.folder != recorded.folder
    assert (latest.folder / 'src' / 'pkg' / '__init__.py').read_text(encoding='utf-8') == (
        '# package v2\n'
    )


def test_new_code_snapshot_and_no_code_snapshot_is_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_backend: FakeBackend
) -> None:
    git_project(tmp_path=tmp_path, monkeypatch=monkeypatch)
    assert (
        run_main(
            'resubmit', 'study', '--machine', 'local', '--new-code-snapshot', '--no-code-snapshot'
        )
        == 2
    )


def test_include_untracked_needs_new_code_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_backend: FakeBackend
) -> None:
    git_project(tmp_path=tmp_path, monkeypatch=monkeypatch)
    assert (
        run_main('resubmit', 'study', '--machine', 'local', '--code-snapshot-include-untracked')
        == 2
    )


def test_code_snapshot_in_toml_does_not_change_resubmit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_backend: FakeBackend
) -> None:
    project = git_project(tmp_path=tmp_path, monkeypatch=monkeypatch, code_snapshot=True)
    # why: --no-code-snapshot beats the toml, so the run is recorded without one
    assert run_main('submit', 'study', '--machine', 'local', '--no-code-snapshot') == 0
    assert jobs_of(project=project)[-1].code_snapshot is None
    _fail_all(project=project, fake_backend=fake_backend)
    assert run_main('resubmit', 'study', '--machine', 'local', '--states', 'failed') == 0
    assert jobs_of(project=project)[-1].code_snapshot is None


def test_missing_folder_exit_2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_backend: FakeBackend,
    caplog: pytest.LogCaptureFixture,
) -> None:
    project = git_project(tmp_path=tmp_path, monkeypatch=monkeypatch)
    assert run_main('submit', 'study', '--machine', 'local', '--code-snapshot') == 0
    recorded = jobs_of(project=project)[-1].code_snapshot
    assert recorded is not None
    shutil.rmtree(recorded.folder)
    _fail_all(project=project, fake_backend=fake_backend)
    with caplog.at_level(logging.ERROR):
        code = run_main('resubmit', 'study', '--machine', 'local', '--states', 'failed')
    assert code == 2
    assert str(recorded.folder) in caplog.text
    assert '--new-code-snapshot' in caplog.text

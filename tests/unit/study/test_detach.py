"""``--detach``: a local launch that keeps running after the command returns."""

from collections.abc import Callable
import logging
from pathlib import Path
import time

import pytest

from tests.unit.study.helpers import context, study_directory
from tslab.cli import main
from tslab.study.local_launch import detach_launch, LAUNCH_FILE, LAUNCH_LOG
from tslab.study.state import StudyState


def _detach(*extra: str, machine: str = 'local.toml') -> int:
    with pytest.raises(SystemExit) as raised:
        main(
            args=['study', 'submit', 'toy_studies:study', '--machine', machine, '--detach', *extra]
        )
    code = raised.value.code
    assert isinstance(code, int)
    return code


def _states(*, project: Path) -> set[StudyState]:
    reports = context(project=project, machine_file='local.toml').reports
    return {report.state for report in reports}


def _pid_file(*, project: Path) -> Path:
    return study_directory(project=project) / '.study' / LAUNCH_FILE


def _wait_until(*, condition: Callable[[], bool], seconds: float = 20.0) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.1)
    pytest.fail('condition not met in time')


@pytest.fixture
def slow_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every toy run sleep for two seconds."""
    monkeypatch.setenv('TOY_SLEEP', '2')


def test_detach_returns_at_once_with_pid_file(project: Path, slow_tasks: None) -> None:
    start = time.monotonic()
    code = _detach()
    elapsed = time.monotonic() - start
    assert code == 0
    assert elapsed < 5
    assert _pid_file(project=project).exists()
    _wait_until(condition=lambda: not _pid_file(project=project).exists())


def test_detach_status_shows_running_and_queued(project: Path, slow_tasks: None) -> None:
    assert _detach() == 0
    states = {
        report.state for report in context(project=project, machine_file='local.toml').reports
    }
    assert StudyState.RUNNING in states
    assert StudyState.QUEUED in states or StudyState.DONE in states
    _wait_until(condition=lambda: not _pid_file(project=project).exists())


def test_wait_ends_after_detached_launch(project: Path, slow_tasks: None) -> None:
    assert _detach() == 0
    with pytest.raises(SystemExit) as raised:
        main(args=['study', 'wait', 'toy_studies:study', '--machine', 'local.toml'])
    assert raised.value.code == 0
    reports = context(project=project, machine_file='local.toml').reports
    assert {report.state for report in reports} == {StudyState.DONE}
    assert not _pid_file(project=project).exists()


def test_second_detach_while_live_fails(project: Path, slow_tasks: None) -> None:
    assert _detach() == 0
    assert _detach() == 1
    _wait_until(condition=lambda: not _pid_file(project=project).exists())


def test_detach_on_slurm_is_usage_error(project: Path) -> None:
    assert _detach(machine='cluster.toml') == 2


def test_detach_reports_a_supervisor_that_fails(
    project: Path, caplog: pytest.LogCaptureFixture
) -> None:
    folder = study_directory(project=project) / '.study'
    start = time.monotonic()
    with caplog.at_level(logging.INFO):
        code = detach_launch(
            study_folder=folder,
            arguments=['study', 'submit', 'no_such_module:study', '--machine', 'local.toml'],
            project_root=project,
            wait_command='tslab study wait no_such_module:study --machine local.toml',
        )
    assert code != 0
    assert time.monotonic() - start < 10
    assert (folder / LAUNCH_LOG).exists()
    assert 'no_such_module' in caplog.text


def test_cancel_detached_launch_interrupts_run(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('TOY_SLEEP', '60')
    assert _detach() == 0
    _wait_until(
        condition=lambda: any(
            report.folder.snapshot is not None
            for report in context(project=project, machine_file='local.toml').reports
        )
    )
    with pytest.raises(SystemExit) as raised:
        main(args=['study', 'cancel', 'toy_studies:study', '--machine', 'local.toml'])
    assert raised.value.code == 0
    _wait_until(condition=lambda: not _pid_file(project=project).exists(), seconds=5)
    states = {
        report.state for report in context(project=project, machine_file='local.toml').reports
    }
    assert StudyState.INTERRUPTED in states

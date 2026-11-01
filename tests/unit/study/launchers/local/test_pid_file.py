"""The local launch pid file, task states and detach behavior."""

from collections.abc import Callable
import logging
import os
from pathlib import Path
import time

import pytest

from tests.unit.study.helpers import (
    context,
    finished_pid,
    record_local_job,
    study_directory,
    write_launch,
)
from tslab.cli import main
from tslab.core.meta import hostname
from tslab.core.paths import LOCK_FILE
from tslab.core.status import (
    InstanceRecord,
    InstanceState,
    RunState,
    RunStatusSnapshot,
    utc_now_iso,
    write_status,
)
from tslab.study.launchers.local.pid_file import (
    detach_launch,
    hold_launch,
    LAUNCH_FILE,
    launch_is_live,
    LAUNCH_LOG,
    local_job_states,
    LocalLaunchError,
    read_launch,
)
from tslab.study.reporting.state import StudyState


def test_hold_launch_writes_and_removes_file(tmp_path: Path) -> None:
    with hold_launch(
        study_folder=tmp_path,
        task_ids=['j_0', 'j_1'],
        pid=os.getpid(),
        host=hostname(),
        now=utc_now_iso(),
    ) as set_state:
        launch = read_launch(study_folder=tmp_path)
        assert launch is not None
        assert launch.pid == os.getpid()
        assert launch.tasks == (('j_0', 'pending'), ('j_1', 'pending'))
        set_state(task_id='j_0', state='running')
        written = read_launch(study_folder=tmp_path)
        assert written is not None
        assert written.tasks == (('j_0', 'running'), ('j_1', 'pending'))
    assert read_launch(study_folder=tmp_path) is None


def test_hold_launch_uses_given_pid_host_and_time(tmp_path: Path) -> None:
    started = '2026-01-02T03:04:05+00:00'

    with hold_launch(
        study_folder=tmp_path, task_ids=['job_0'], pid=123, host='test-host', now=started
    ):
        launch = read_launch(study_folder=tmp_path)

    assert launch is not None
    assert (launch.pid, launch.hostname, launch.started) == (123, 'test-host', started)


def test_hold_launch_refuses_live_launch(tmp_path: Path) -> None:
    write_launch(folder=tmp_path, pid=os.getppid(), host=hostname(), tasks={})
    with (
        pytest.raises(LocalLaunchError, match=rf'{os.getppid()}.*tslab study cancel'),
        hold_launch(
            study_folder=tmp_path, task_ids=[], pid=os.getpid(), host=hostname(), now=utc_now_iso()
        ),
    ):
        pytest.fail('must not enter')


def test_hold_launch_removes_stale_file_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    write_launch(folder=tmp_path, pid=finished_pid(), host=hostname(), tasks={'old_0': 'running'})
    with (
        caplog.at_level('WARNING'),
        hold_launch(
            study_folder=tmp_path,
            task_ids=['j_0'],
            pid=os.getpid(),
            host=hostname(),
            now=utc_now_iso(),
        ),
    ):
        launch = read_launch(study_folder=tmp_path)
        assert launch is not None
        assert launch.pid == os.getpid()
    assert caplog.text.count('stale') == 1


def test_hold_launch_refuses_other_host(tmp_path: Path) -> None:
    write_launch(folder=tmp_path, pid=1, host='elsewhere', tasks={})
    with (
        pytest.raises(LocalLaunchError, match=rf'elsewhere.*{LAUNCH_FILE}'),
        hold_launch(
            study_folder=tmp_path, task_ids=[], pid=os.getpid(), host=hostname(), now=utc_now_iso()
        ),
    ):
        pytest.fail('must not enter')
    assert (tmp_path / LAUNCH_FILE).exists()


def test_local_job_states_map_pending_and_running(tmp_path: Path) -> None:
    write_launch(
        folder=tmp_path,
        pid=os.getppid(),
        host=hostname(),
        tasks={'j_0': 'pending', 'j_1': 'running', 'j_2': 'ended'},
    )
    assert local_job_states(study_folder=tmp_path, host=hostname()) == {
        'j_0': 'pending',
        'j_1': 'running',
        'j_2': 'ended',
    }
    launch = read_launch(study_folder=tmp_path)
    assert launch is not None
    assert launch_is_live(launch=launch, host=hostname())


def test_local_job_states_without_live_launch_is_empty(tmp_path: Path) -> None:
    write_launch(folder=tmp_path, pid=finished_pid(), host=hostname(), tasks={'j_0': 'running'})
    assert local_job_states(study_folder=tmp_path, host=hostname()) == {}


def test_status_reads_local_launch_states(project: Path) -> None:
    record_local_job(project=project)
    write_launch(
        folder=study_directory(project=project) / '.study',
        pid=os.getppid(),
        host=hostname(),
        tasks={'local-1-1_0': 'running', 'local-1-1_1': 'pending'},
    )
    states = {
        report.run.run_path: report.state
        for report in context(project=project, machine_file='local.toml').reports
    }
    paths = list(states)
    assert states[paths[0]] is StudyState.RUNNING
    assert states[paths[1]] is StudyState.QUEUED


def test_run_left_running_without_launch_reads_interrupted(project: Path) -> None:
    record_local_job(project=project)
    ctx = context(project=project, machine_file='local.toml')
    folder = study_directory(project=project) / ctx.runs[0].run_path
    folder.mkdir(parents=True)
    write_status(
        path=folder / 'status.json',
        snapshot=RunStatusSnapshot(
            run_state=RunState.RUNNING,
            updated_at=utc_now_iso(),
            instances={'train': InstanceRecord(state=InstanceState.RUNNING)},
        ),
    )
    # a fresh heartbeat held by a pid that is not a live launch
    (folder / LOCK_FILE).write_text(
        f'{{"pid": {os.getpid()}, "host": "{hostname()}", "nonce": "n"}}', encoding='utf-8'
    )
    reports = context(project=project, machine_file='local.toml').reports
    assert reports[0].state is StudyState.INTERRUPTED
    assert reports[0].reason is None


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

"""The local launch pid file, task states and detach behavior."""

from collections.abc import Callable
import contextlib
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from tests.lab.unit.study.helpers import (
    context,
    finished_pid,
    record_local_job,
    study_directory,
    write_launch,
)
from tslab.lab.cli import main
from tslab.lab.core.files import atomic_write_json
from tslab.lab.core.meta import hostname
from tslab.lab.core.paths import LOCK_FILE
from tslab.lab.core.status import InstanceRecord, RunStatusSnapshot, utc_now_iso, write_status
from tslab.lab.enums import InstanceState, LocalTaskState, RunState, StudyState
from tslab.lab.jobs.local import process_alive
from tslab.lab.study.launchers.local import pid_file
from tslab.lab.study.launchers.local.pid_file import (
    _stop_detached,
    detach_launch,
    hold_launch,
    LAUNCH_FILE,
    launch_is_live,
    LAUNCH_LOGS,
    local_job_states,
    LocalLaunch,
    LocalLaunchError,
    read_launch,
)


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
        set_state(task_id='j_0', state=LocalTaskState.RUNNING)
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


def test_launch_from_another_boot_is_stale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pid_file, 'boot_id', lambda: 'boot-B')
    launch = LocalLaunch(pid=os.getpid(), hostname='h', started='t', tasks=(), boot='boot-A')
    assert launch_is_live(launch=launch, host='h') is False


def test_launch_from_this_boot_with_a_live_pid_is_live(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pid_file, 'boot_id', lambda: 'boot-A')
    launch = LocalLaunch(pid=os.getpid(), hostname='h', started='t', tasks=(), boot='boot-A')
    assert launch_is_live(launch=launch, host='h') is True


def test_old_launch_file_without_boot_uses_the_pid(tmp_path: Path) -> None:
    write_launch(folder=tmp_path, pid=os.getpid(), host='h', tasks={})
    launch = read_launch(study_folder=tmp_path)
    assert launch is not None
    assert launch.boot is None
    assert launch_is_live(launch=launch, host='h') is True


def test_hold_launch_writes_the_boot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pid_file, 'boot_id', lambda: 'boot-A')
    with hold_launch(study_folder=tmp_path, task_ids=['x'], pid=1, host='h', now='t'):
        launch = read_launch(study_folder=tmp_path)
        assert launch is not None
        assert launch.boot == 'boot-A'


def test_hold_launch_records_run_pids(tmp_path: Path) -> None:
    with hold_launch(
        study_folder=tmp_path, task_ids=['j_0', 'j_1'], pid=1, host='h', now='t'
    ) as set_state:
        set_state(task_id='j_0', state=LocalTaskState.RUNNING, pid=4242)
        launch = read_launch(study_folder=tmp_path)
        assert launch is not None
        assert launch.run_pids == (('j_0', 4242),)
        set_state(task_id='j_0', state=LocalTaskState.ENDED)
        launch = read_launch(study_folder=tmp_path)
        assert launch is not None
        assert launch.run_pids == ()


def test_old_launch_file_has_no_run_pids(tmp_path: Path) -> None:
    write_launch(folder=tmp_path, pid=os.getpid(), host='h', tasks={'j_0': 'running'})
    launch = read_launch(study_folder=tmp_path)
    assert launch is not None
    assert launch.run_pids == ()


def test_run_pids_are_written_as_an_object(tmp_path: Path) -> None:
    with hold_launch(
        study_folder=tmp_path, task_ids=['j_0', 'j_1'], pid=1, host='h', now='t'
    ) as set_state:
        set_state(task_id='j_0', state=LocalTaskState.RUNNING, pid=4242)
        document = json.loads((tmp_path / LAUNCH_FILE).read_text())
    assert document['run_pids'] == {'j_0': 4242}


def test_scalar_run_pids_are_rejected(tmp_path: Path) -> None:
    write_launch(folder=tmp_path, pid=os.getpid(), host='h', tasks={'j_0': 'running'})
    path = tmp_path / LAUNCH_FILE
    document = json.loads(path.read_text())
    document['run_pids'] = 4242
    atomic_write_json(path=path, document=document)
    with pytest.raises(LocalLaunchError, match=r'not a local launch file'):
        read_launch(study_folder=tmp_path)


def test_a_launch_file_of_an_older_build_names_the_file_and_the_way_out(tmp_path: Path) -> None:
    write_launch(folder=tmp_path, pid=os.getpid(), host='h', tasks={'j_0': 'running'})
    path = tmp_path / LAUNCH_FILE
    document = json.loads(path.read_text())
    document['run_pids'] = [['j_0', 4242]]
    atomic_write_json(path=path, document=document)
    with pytest.raises(LocalLaunchError, match=r'older build.*no longer running.*remove') as info:
        read_launch(study_folder=tmp_path)
    assert str(path) in str(info.value)


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
    logs = list((folder / LAUNCH_LOGS).iterdir())
    assert [path.name for path in logs] != []
    assert all(not path.name.startswith('starting-') for path in logs)
    assert 'no_such_module' in caplog.text


def test_detach_stops_a_launch_that_does_not_start(
    project: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    started: list[subprocess.Popen[bytes]] = []

    def fake_start(**_kwargs: object) -> subprocess.Popen[bytes]:
        started.append(
            subprocess.Popen(
                [sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True
            )
        )
        return started[0]

    monkeypatch.setattr(pid_file, 'start_detached', fake_start)
    folder = study_directory(project=project) / '.study'
    try:
        with caplog.at_level(logging.ERROR):
            code = detach_launch(
                study_folder=folder,
                arguments=[],
                project_root=project,
                wait_command='w',
                timeout_seconds=0.5,
            )
        assert code == 1
        assert started[0].poll() is not None
        assert 'stopped' in caplog.text
    finally:
        process = started[0] if started else None
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()


def test_detach_logs_a_clean_early_exit_as_info(
    project: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(
        pid_file,
        'start_detached',
        lambda **_kwargs: subprocess.Popen([sys.executable, '-c', 'pass'], start_new_session=True),
    )
    folder = study_directory(project=project) / '.study'
    with caplog.at_level(logging.INFO):
        code = detach_launch(
            study_folder=folder, arguments=[], project_root=project, wait_command='w'
        )
    assert code == 0
    assert [record.levelno for record in caplog.records] == [logging.INFO]
    assert 'exit code 0' in caplog.text


def test_detach_stop_ignores_a_child_that_already_exited() -> None:
    process = subprocess.Popen([sys.executable, '-c', 'pass'], start_new_session=True)
    process.wait()

    _stop_detached(process=process)

    assert process.poll() == 0


@pytest.mark.slow
def test_detach_kills_a_child_that_ignores_sigterm() -> None:
    child_code = (
        'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); '
        "print('ready', flush=True); time.sleep(60)"
    )
    leader_code = (
        'import subprocess, sys, time; '
        f"child = subprocess.Popen([sys.executable, '-c', {child_code!r}], "
        'stdout=subprocess.PIPE); '
        'print(child.pid, flush=True); '
        'print(child.stdout.readline().decode(), flush=True); '
        'time.sleep(60)'
    )
    process = subprocess.Popen(  # noqa: S603 # why: fixed test command
        [sys.executable, '-c', leader_code], stdout=subprocess.PIPE, start_new_session=True
    )
    try:
        assert process.stdout is not None
        child_pid = int(process.stdout.readline())
        assert process.stdout.readline() == b'ready\n'

        _stop_detached(process=process)

        assert process.poll() is not None
        _wait_until(condition=lambda: not process_alive(pid=child_pid), seconds=2)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


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


def test_read_launch_rejects_an_unknown_task_state(tmp_path: Path) -> None:
    write_launch(folder=tmp_path, pid=1, host=hostname(), tasks={'j_0': 'paused'})
    with pytest.raises(LocalLaunchError, match="'paused' is not a valid LocalTaskState"):
        read_launch(study_folder=tmp_path)

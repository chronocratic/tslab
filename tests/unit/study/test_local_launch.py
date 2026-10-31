"""The pid file of a local launch and the job states read from it."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

from tests.unit.study.helpers import context, study_directory
from tslab.core.paths import atomic_write_json, LOCK_FILE
from tslab.core.status import (
    InstanceRecord,
    InstanceState,
    RunState,
    RunStatusSnapshot,
    utc_now_iso,
    write_status,
)
from tslab.study.local_launch import (
    hold_launch,
    LAUNCH_FILE,
    launch_is_live,
    local_job_states,
    LocalLaunchError,
    read_launch,
)
from tslab.study.records import append_job, JobRecord, ProgressSnapshot, TaskRecord
from tslab.study.state import LOCAL_ENDED, StudyState


def write_launch(*, folder: Path, pid: int, host: str, tasks: dict[str, str]) -> None:
    """Write a launch file as ``hold_launch`` would."""
    folder.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        path=folder / LAUNCH_FILE,
        document={
            'pid': pid,
            'hostname': host,
            'started': '2026-01-01T00:00:00+00:00',
            'tasks': [[key, state] for key, state in tasks.items()],
        },
    )


def finished_pid() -> int:
    """The pid of a child process that has already exited."""
    child = subprocess.Popen([sys.executable, '-c', 'pass'])  # why: fixed command
    child.wait()
    return child.pid


def test_hold_launch_writes_and_removes_file(tmp_path: Path) -> None:
    with hold_launch(study_folder=tmp_path, task_ids=['j_0', 'j_1']) as set_state:
        launch = read_launch(study_folder=tmp_path)
        assert launch is not None
        assert launch.pid == os.getpid()
        assert launch.tasks == (('j_0', 'pending'), ('j_1', 'pending'))
        set_state('j_0', 'running')
        written = read_launch(study_folder=tmp_path)
        assert written is not None
        assert written.tasks == (('j_0', 'running'), ('j_1', 'pending'))
    assert read_launch(study_folder=tmp_path) is None


def test_hold_launch_refuses_live_launch(tmp_path: Path) -> None:
    write_launch(folder=tmp_path, pid=os.getppid(), host=_host(), tasks={})
    with (
        pytest.raises(LocalLaunchError, match=rf'{os.getppid()}.*tslab study cancel'),
        hold_launch(study_folder=tmp_path, task_ids=[]),
    ):
        pytest.fail('must not enter')


def test_hold_launch_removes_stale_file_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    write_launch(folder=tmp_path, pid=finished_pid(), host=_host(), tasks={'old_0': 'running'})
    with caplog.at_level('WARNING'), hold_launch(study_folder=tmp_path, task_ids=['j_0']):
        launch = read_launch(study_folder=tmp_path)
        assert launch is not None
        assert launch.pid == os.getpid()
    assert caplog.text.count('stale') == 1


def test_hold_launch_refuses_other_host(tmp_path: Path) -> None:
    write_launch(folder=tmp_path, pid=1, host='elsewhere', tasks={})
    with (
        pytest.raises(LocalLaunchError, match=rf'elsewhere.*{LAUNCH_FILE}'),
        hold_launch(study_folder=tmp_path, task_ids=[]),
    ):
        pytest.fail('must not enter')
    assert (tmp_path / LAUNCH_FILE).exists()


def test_local_job_states_map_pending_and_running(tmp_path: Path) -> None:
    write_launch(
        folder=tmp_path,
        pid=os.getppid(),
        host=_host(),
        tasks={'j_0': 'pending', 'j_1': 'running', 'j_2': 'ended'},
    )
    assert local_job_states(study_folder=tmp_path) == {
        'j_0': 'PENDING',
        'j_1': 'RUNNING',
        'j_2': LOCAL_ENDED,
    }
    launch = read_launch(study_folder=tmp_path)
    assert launch is not None
    assert launch_is_live(launch=launch)


def test_local_job_states_without_live_launch_is_empty(tmp_path: Path) -> None:
    write_launch(folder=tmp_path, pid=finished_pid(), host=_host(), tasks={'j_0': 'running'})
    assert local_job_states(study_folder=tmp_path) == {}


def _host() -> str:
    from tslab.core.meta import hostname  # noqa: PLC0415 # why: keeps the import list short

    return hostname()


def local_machine(*, project: Path) -> str:
    """Write ``local.toml`` and return its name."""
    (project / 'local.toml').write_text(
        f'launcher = "local"\noutput_root = "{(project / "runs").as_posix()}"\n'
    )
    return 'local.toml'


def record_local_job(*, project: Path) -> list[str]:
    """Record one local job holding every run of the toy study; return the task ids."""
    ctx = context(project=project, machine_file='local.toml')
    tasks = tuple(
        TaskRecord(
            index=index,
            grid=run.grid,
            run_path=run.run_path,
            command=('true',),
            extra_args=(),
            resources={},
            attempt=1,
            progress=ProgressSnapshot(done_instances=0, newest_checkpoint_mtime=None),
        )
        for index, run in enumerate(ctx.runs)
    )
    append_job(
        path=study_directory(project=project) / '.study' / 'submissions.jsonl',
        record=JobRecord(
            kind='run',
            submission_id='s1',
            time=utc_now_iso(),
            launcher='local',
            job_id='local-1-1',
            job_file=1,
            tasks=tasks,
        ),
    )
    return [f'local-1-{task.index}' for task in tasks]


def test_status_reads_local_launch_states(project: Path) -> None:
    local_machine(project=project)
    record_local_job(project=project)
    write_launch(
        folder=study_directory(project=project) / '.study',
        pid=os.getppid(),
        host=_host(),
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
    local_machine(project=project)
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
        f'{{"pid": {os.getpid()}, "host": "{_host()}", "nonce": "n"}}', encoding='utf-8'
    )
    reports = context(project=project, machine_file='local.toml').reports
    assert reports[0].state is StudyState.INTERRUPTED
    assert reports[0].reason is None

"""Cancelling a live local study launch."""

import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from tests.support.fake_slurm import FakeSlurm
from tests.unit.study.helpers import (
    context,
    no_sleep,
    record_local_job,
    study_directory,
    write_launch,
)
from tslab.core.meta import hostname
from tslab.study.commands import cancel, resubmit, submit
from tslab.study.context import load_context, Selection


def _sleeper() -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, '-c', 'import time; time.sleep(30)']
    )  # why: fixed command


def _hold_live_launch(*, project: Path, pid: int, running: tuple[str, ...]) -> None:
    ids = record_local_job(project=project)
    write_launch(
        folder=study_directory(project=project) / '.study',
        pid=pid,
        host=hostname(),
        tasks=dict.fromkeys(ids, 'pending') | dict.fromkeys(running, 'running'),
    )


def test_local_cancel_without_launch_is_noop(
    project: Path, caplog: pytest.LogCaptureFixture
) -> None:
    ctx = context(project=project, machine_file='local.toml')
    with caplog.at_level('INFO', logger='tslab.study.launchers.local.cancel'):
        assert cancel(context=ctx, dry_run=False) == (0, [])
    assert 'nothing to cancel' in caplog.text


def test_local_cancel_dry_run_sends_nothing(project: Path) -> None:
    child = _sleeper()
    try:
        _hold_live_launch(project=project, pid=child.pid, running=())
        ctx = context(project=project, machine_file='local.toml')
        assert cancel(context=ctx, dry_run=True) == (0, [str(child.pid)])
        assert child.poll() is None
    finally:
        child.kill()
        child.wait()


def test_local_cancel_signals_supervisor(project: Path, tmp_path: Path) -> None:
    marker = tmp_path / 'term'
    code = (
        'import signal, sys, time\n'
        'def stop(*_):\n'
        f'    open({str(marker)!r}, "w").write("term")\n'
        '    sys.exit(0)\n'
        'signal.signal(signal.SIGTERM, stop)\n'
        'print("ready", flush=True)\n'
        'time.sleep(30)\n'
    )
    child = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE)  # noqa: S603 # why: fixed command
    try:
        assert child.stdout is not None
        child.stdout.readline()  # the handler is installed once it prints
        _hold_live_launch(project=project, pid=child.pid, running=())
        ctx = context(project=project, machine_file='local.toml')
        assert cancel(context=ctx, dry_run=False) == (0, [str(child.pid)])
        deadline = time.monotonic() + 3
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert marker.read_text() == 'term'
    finally:
        child.kill()
        child.wait()


def test_local_cancel_ignores_tasks_of_groups_not_started(project: Path) -> None:
    child = _sleeper()
    try:
        ids = record_local_job(project=project)
        write_launch(
            folder=study_directory(project=project) / '.study',
            pid=child.pid,
            host=hostname(),
            tasks=dict.fromkeys(ids, 'pending') | {'local-1-9_0': 'pending'},
        )
        ctx = context(project=project, machine_file='local.toml')
        assert cancel(context=ctx, dry_run=True) == (0, [str(child.pid)])
    finally:
        child.kill()
        child.wait()


def test_local_cancel_signal_flag_is_usage_error(
    project: Path, caplog: pytest.LogCaptureFixture
) -> None:
    ctx = context(project=project, machine_file='local.toml')
    with caplog.at_level('ERROR', logger='tslab.study.launchers.local.cancel'):
        assert cancel(context=ctx, dry_run=False, signal='USR2') == (2, [])
    assert 'cancel --signal is SLURM only' in caplog.text


def test_local_cancel_partial_selection_is_usage_error(
    project: Path, caplog: pytest.LogCaptureFixture
) -> None:
    child = _sleeper()
    try:
        _hold_live_launch(project=project, pid=child.pid, running=())
        ctx = load_context(
            reference='toy_studies:study',
            machine_path=project / 'local.toml',
            selection=Selection(only=('seed=1',)),
            project_root=Path.cwd(),
            local_cpus=1,
            environment=os.environ,
        )
        with caplog.at_level('ERROR', logger='tslab.study.launchers.local.cancel'):
            assert cancel(context=ctx, dry_run=False) == (2, [])
        assert 'as one unit' in caplog.text
        assert child.poll() is None
    finally:
        child.kill()
        child.wait()


@pytest.mark.slow
def test_cancel_live_never_cancels_local_jobs(project: Path, fake: FakeSlurm) -> None:
    # why: local job ids (local-<pid>-<n>) are not SLURM jobs
    code = submit(
        context=context(project=project, machine_file='local.toml'),
        include_done=False,
        dry_run=False,
        extra_args=(),
        sleep=no_sleep,
    )
    assert code == 0
    (project / 'runs' / 's' / 'g/small/1' / '.lock').write_text('')
    code = resubmit(
        context=context(project=project, machine_file='local.toml'),
        cancel_live=True,
        sleep=no_sleep,
    )
    assert code == 0
    assert fake.calls('scancel') == []

"""Cancelling a live local study launch."""

import contextlib
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
    no_sleep,
    record_local_job,
    study_directory,
    write_launch,
)
from tests.support.fake_slurm import FakeSlurm
from tslab.lab.cli import main
from tslab.lab.core.meta import hostname
from tslab.lab.jobs.local import boot_id
from tslab.lab.study.commands import cancel, resubmit, submit
from tslab.lab.study.context import load_context, Selection


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
    with caplog.at_level('INFO', logger='tslab.lab.study.launchers.local.cancel'):
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
        while time.monotonic() < deadline and (not marker.exists() or marker.read_text() != 'term'):
            time.sleep(0.05)
        # why: the handler creates the marker before buffered text is visible.
        assert marker.exists()
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


def _run_group() -> subprocess.Popen[bytes]:
    return subprocess.Popen(  # why: a fixed test command
        [sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True
    )


def _hold_dead_launch(*, project: Path, run_pid: int) -> list[str]:
    ids = record_local_job(project=project)
    write_launch(
        folder=study_directory(project=project) / '.study',
        pid=finished_pid(),
        host=hostname(),
        tasks=dict.fromkeys(ids, 'pending') | {ids[0]: 'running'},
        run_pids=((ids[0], run_pid),),
        boot=boot_id(),
    )
    return ids


def test_cancel_signals_runs_of_a_dead_launcher(project: Path) -> None:
    run = _run_group()
    try:
        _hold_dead_launch(project=project, run_pid=run.pid)
        ctx = context(project=project, machine_file='local.toml')
        assert cancel(context=ctx, dry_run=False) == (0, [str(run.pid)])
        deadline = time.monotonic() + 5
        while run.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert run.poll() is not None
    finally:
        if run.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                run.kill()
        run.wait()


def test_cancel_dry_run_lists_runs_of_a_dead_launcher(project: Path) -> None:
    run = _run_group()
    try:
        _hold_dead_launch(project=project, run_pid=run.pid)
        ctx = context(project=project, machine_file='local.toml')
        assert cancel(context=ctx, dry_run=True) == (0, [str(run.pid)])
        assert run.poll() is None
    finally:
        if run.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                run.kill()
        run.wait()


def test_local_cancel_dead_partial_selection_is_usage_error(
    project: Path, caplog: pytest.LogCaptureFixture
) -> None:
    run = _run_group()
    try:
        ids = _hold_dead_launch(project=project, run_pid=run.pid)
        ctx = load_context(
            reference='toy_studies:study',
            machine_path=project / 'local.toml',
            selection=Selection(only=('seed=1',)),
            project_root=Path.cwd(),
            local_cpus=1,
            environment=os.environ,
        )
        selected = {report.job_id for report in ctx.reports}
        missed = [key for key in ids if key not in selected]
        assert missed
        with caplog.at_level('ERROR', logger='tslab.lab.study.launchers.local.cancel'):
            assert cancel(context=ctx, dry_run=False) == (2, [])
        assert 'as one unit' in caplog.text
        assert all(key in caplog.text for key in missed)
        assert run.poll() is None
    finally:
        if run.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                run.kill()
        run.wait()


def test_cancel_dead_keeps_signalling_when_a_group_gone(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _run_group()
    second = _run_group()
    try:
        ids = record_local_job(project=project)
        write_launch(
            folder=study_directory(project=project) / '.study',
            pid=finished_pid(),
            host=hostname(),
            tasks=dict.fromkeys(ids, 'pending') | {ids[0]: 'running', ids[1]: 'running'},
            run_pids=((ids[0], first.pid), (ids[1], second.pid)),
            boot=boot_id(),
        )
        real_killpg = os.killpg

        def killpg(pid: int, sig: int) -> None:
            # why: the first group dies between the existence check and the signal
            if pid == first.pid and sig == signal.SIGTERM:
                raise ProcessLookupError
            real_killpg(pid, sig)

        monkeypatch.setattr(os, 'killpg', killpg)
        ctx = context(project=project, machine_file='local.toml')
        assert cancel(context=ctx, dry_run=False) == (0, [str(first.pid), str(second.pid)])
        deadline = time.monotonic() + 5
        while second.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert second.poll() is not None
    finally:
        for proc in (first, second):
            if proc.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
            proc.wait()


def test_local_cancel_signal_flag_is_usage_error(
    project: Path, caplog: pytest.LogCaptureFixture
) -> None:
    ctx = context(project=project, machine_file='local.toml')
    with caplog.at_level('ERROR', logger='tslab.lab.study.launchers.local.cancel'):
        assert cancel(context=ctx, dry_run=False, signal='USR2') == (2, [])
    assert 'cancel --signal is SLURM only' in caplog.text


def test_local_cancel_signal_through_the_cli_exits_two(
    project: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level('ERROR'), pytest.raises(SystemExit) as raised:
        main(args=['study', 'cancel', 'toy_studies:study', '--machine', 'local.toml', '--signal'])
    assert raised.value.code == 2
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
        with caplog.at_level('ERROR', logger='tslab.lab.study.launchers.local.cancel'):
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

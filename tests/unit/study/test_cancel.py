"""Cancelling live SLURM runs."""

import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

from tests.support.fake_slurm import FakeSlurm
from tests.unit.study.helpers import (
    context,
    record_local_job,
    study_directory,
    submit_study,
    write_launch,
)
from tslab.cli import main
from tslab.core.meta import hostname
from tslab.study.commands import cancel
from tslab.study.context import Context, load_context, Selection


def _context_with_live_runs(*, project: Path, fake: FakeSlurm) -> Context:
    submit_study(project=project)
    fake.set_answers('squeue', [{'exit': 0, 'stdout': '1000_0|RUNNING\n1000_1|PENDING\n'}])
    fake.set_answers('sacct', [{'exit': 0, 'stdout': '1000_2|COMPLETED\n1000_3|COMPLETED\n'}])
    return context(project=project, machine_file='cluster.toml')


def test_cancel_scancels_live_runs_only(project: Path, fake: FakeSlurm) -> None:
    ctx = _context_with_live_runs(project=project, fake=fake)
    code, cancelled = cancel(context=ctx, dry_run=False)
    assert code == 0
    assert cancelled == ['1000_0', '1000_1']
    assert fake.calls('scancel') == [['1000_0', '1000_1']]


def test_cancel_dry_run_calls_nothing(project: Path, fake: FakeSlurm) -> None:
    ctx = _context_with_live_runs(project=project, fake=fake)
    code, cancelled = cancel(context=ctx, dry_run=True)
    assert code == 0
    assert cancelled == ['1000_0', '1000_1']
    assert fake.calls('scancel') == []


def test_cancel_signal_signals_running_runs_only(project: Path, fake: FakeSlurm) -> None:
    ctx = _context_with_live_runs(project=project, fake=fake)
    code, ids = cancel(context=ctx, dry_run=False, signal='USR2')
    assert (code, ids) == (0, ['1000_0'])
    assert fake.calls('scancel') == [['--signal=USR2', '1000_0']]


def test_cancel_signal_skips_queued_runs_with_a_log(
    project: Path, fake: FakeSlurm, caplog: pytest.LogCaptureFixture
) -> None:
    ctx = _context_with_live_runs(project=project, fake=fake)
    with caplog.at_level('INFO', logger='tslab.study.commands'):
        cancel(context=ctx, dry_run=False, signal='USR2')
    assert 'skipped queued runs: 1000_1' in caplog.text


def test_cancel_signal_dry_run_calls_nothing(project: Path, fake: FakeSlurm) -> None:
    ctx = _context_with_live_runs(project=project, fake=fake)
    code, ids = cancel(context=ctx, dry_run=True, signal='USR2')
    assert (code, ids) == (0, ['1000_0'])
    assert fake.calls('scancel') == []


def test_cancel_signal_cli_json_has_signal(
    project: Path, fake: FakeSlurm, capsys: pytest.CaptureFixture[str]
) -> None:
    _context_with_live_runs(project=project, fake=fake)
    with pytest.raises(SystemExit) as raised:
        main(
            args=[
                'study',
                'cancel',
                'toy_studies:study',
                '--machine',
                'cluster.toml',
                '--signal',
                '--json',
            ]
        )
    document = json.loads(capsys.readouterr().out)
    assert raised.value.code == 0
    assert document['signal'] == 'USR2'
    assert document['cancelled'] == ['1000_0']


def test_cancel_respects_only_selection(project: Path, fake: FakeSlurm) -> None:
    _context_with_live_runs(project=project, fake=fake)
    with pytest.raises(SystemExit) as raised:
        main(
            args=[
                'study',
                'cancel',
                'toy_studies:study',
                '--machine',
                'cluster.toml',
                '--only',
                'model=small',
            ]
        )
    assert raised.value.code == 0
    assert fake.calls('scancel') == [['1000_0', '1000_1']]


def test_cancel_json_shape(
    project: Path, fake: FakeSlurm, capsys: pytest.CaptureFixture[str]
) -> None:
    _context_with_live_runs(project=project, fake=fake)
    with pytest.raises(SystemExit) as raised:
        main(
            args=[
                'study',
                'cancel',
                'toy_studies:study',
                '--machine',
                'cluster.toml',
                '--dry-run',
                '--json',
            ]
        )
    document = json.loads(capsys.readouterr().out)
    assert raised.value.code == 0
    assert document == {
        'schema': 'tslab.study-cancel/1',
        'cancelled': ['1000_0', '1000_1'],
        'dry_run': True,
        'signal': None,
        'exit_code': 0,
    }


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
    with caplog.at_level('INFO', logger='tslab.study.commands'):
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


def test_local_cancel_signal_flag_is_usage_error(
    project: Path, caplog: pytest.LogCaptureFixture
) -> None:
    ctx = context(project=project, machine_file='local.toml')
    with caplog.at_level('ERROR', logger='tslab.study.commands'):
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
        )
        with caplog.at_level('ERROR', logger='tslab.study.commands'):
            assert cancel(context=ctx, dry_run=False) == (2, [])
        assert 'as one unit' in caplog.text
        assert child.poll() is None
    finally:
        child.kill()
        child.wait()


def test_local_cancel_signal_flag_cli_exits_2(project: Path) -> None:
    with pytest.raises(SystemExit) as raised:
        main(args=['study', 'cancel', 'toy_studies:study', '--machine', 'local.toml', '--signal'])
    assert raised.value.code == 2

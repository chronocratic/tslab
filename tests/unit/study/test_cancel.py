"""Cancelling live SLURM runs."""

import json
from pathlib import Path

import pytest

from tests.unit.study.fake_slurm import FakeSlurm
from tests.unit.study.helpers import context, submit_study
from tslab.cli import main
from tslab.study.commands import cancel
from tslab.study.context import Context


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


def test_cancel_on_local_machine_exits_1(
    project: Path, fake: FakeSlurm, caplog: pytest.LogCaptureFixture
) -> None:
    ctx = context(project=project, machine_file='local.toml')
    with caplog.at_level('ERROR', logger='tslab.study.commands'):
        code, cancelled = cancel(context=ctx, dry_run=False)
    assert code == 1
    assert cancelled == []
    assert (
        'local runs run in the foreground of tslab study submit; stop that process' in caplog.text
    )


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
        'exit_code': 0,
    }

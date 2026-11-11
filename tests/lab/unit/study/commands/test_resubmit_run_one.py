"""``tslab study run``: one run in the foreground, nothing recorded."""

from collections.abc import Callable, Sequence
import dataclasses
import os
from pathlib import Path
import shlex
from typing import cast

import pytest

from tests.lab.unit.study.helpers import submit_study
from tests.support.fake_backend import FakeBackend
from tslab.lab.cli.entrypoint import main
from tslab.lab.enums import JobPhase
from tslab.lab.study.commands.resubmit import run_one
from tslab.lab.study.context import Context, load_context, Selection


def _one_run_context(*, project: Path, machine_file: str, only: tuple[str, ...]) -> Context:
    return load_context(
        reference='toy_studies:study',
        machine_path=project / machine_file,
        selection=Selection(only=only),
        project_root=Path.cwd(),
        local_cpus=1,
        environment=os.environ,
    )


def _process(*, code: int = 3) -> tuple[list[list[str]], Callable[[list[str]], int]]:
    calls: list[list[str]] = []

    def run_process(args: list[str]) -> int:
        calls.append(args)
        return code

    return calls, run_process


def test_run_one_runs_the_command_of_one_run(project: Path, fake_backend: FakeBackend) -> None:
    submit_study(project=project)
    ctx = _one_run_context(
        project=project, machine_file='cluster.toml', only=('model=small', 'seed=1')
    )
    calls, process = _process()
    assert run_one(context=ctx, extra_args=None, run_process=process) == 3
    assert len(calls) == 1
    args = calls[0]
    assert args[:2] == ['bash', '-c']
    last = args[2].strip().splitlines()[-1]
    assert last.startswith('exec ')
    assert 'toy_script.py' in last
    assert 'toy_script.py' in shlex.join(
        cast('Sequence[str]', fake_backend.calls('foreground_script')[0]['command'])
    )


def test_run_one_reuses_extra_args_of_latest_submission(
    project: Path, fake_backend: FakeBackend, tmp_path: Path
) -> None:
    track = tmp_path / 'track.log'
    submit_study(project=project, extra_args=('--tracking-log', str(track)))
    ctx = _one_run_context(
        project=project, machine_file='cluster.toml', only=('model=small', 'seed=1')
    )
    calls, process = _process()
    run_one(context=ctx, extra_args=None, run_process=process)
    assert calls[0][2].rstrip().endswith(shlex.join(('--tracking-log', str(track))))


def test_run_one_args_after_dashes_replace_them(
    project: Path, fake_backend: FakeBackend, tmp_path: Path
) -> None:
    track = tmp_path / 'track.log'
    submit_study(project=project, extra_args=('--tracking-log', str(track)))
    other = tmp_path / 'other.log'
    ctx = _one_run_context(
        project=project, machine_file='cluster.toml', only=('model=small', 'seed=1')
    )
    calls, process = _process()
    run_one(context=ctx, extra_args=('--tracking-log', str(other)), run_process=process)
    assert calls[0][2].rstrip().endswith(shlex.join(('--tracking-log', str(other))))
    assert shlex.join(('--tracking-log', str(track))) not in calls[0][2]


def test_run_one_refuses_live_run(
    project: Path, fake_backend: FakeBackend, caplog: pytest.LogCaptureFixture
) -> None:
    submit_study(project=project)
    fake_backend.set_states(phases={'job-0_0': JobPhase.RUNNING})
    ctx = _one_run_context(
        project=project, machine_file='cluster.toml', only=('model=small', 'seed=1')
    )
    calls, process = _process()
    with caplog.at_level('ERROR', logger='tslab.lab.study.commands'):
        assert run_one(context=ctx, extra_args=None, run_process=process) == 1
    assert calls == []
    assert 'g/small/1 is running (job job-0_0); wait for it or cancel it first' in caplog.text


def test_run_one_without_a_selection_raises(project: Path, fake_backend: FakeBackend) -> None:
    submit_study(project=project)
    ctx = _one_run_context(project=project, machine_file='cluster.toml', only=('model=small',))
    empty = dataclasses.replace(ctx, reports=())
    with pytest.raises(ValueError, match='no runs match the selection'):
        run_one(context=empty, extra_args=None)


def test_run_one_cli_lists_more_than_one_run(
    project: Path, fake_backend: FakeBackend, caplog: pytest.LogCaptureFixture
) -> None:
    submit_study(project=project)
    with caplog.at_level('ERROR', logger='tslab.lab.cli'), pytest.raises(SystemExit) as raised:
        main(args=['study', 'run', 'toy_studies:study', '--machine', 'cluster.toml'])
    assert raised.value.code == 2
    text = caplog.text
    assert '4 runs match the selection' in text
    assert 'g/small/1' in text
    assert 'g/large/2' in text
    assert 'narrow with --only' in text


def test_run_one_records_nothing(project: Path, fake_backend: FakeBackend) -> None:
    submit_study(project=project)
    study = project / 'runs' / 's' / '.study'
    before = (study / 'submissions.jsonl').read_text()
    jobs_before = sorted(path.name for path in (study / 'jobs').glob('*'))
    ctx = _one_run_context(
        project=project, machine_file='cluster.toml', only=('model=small', 'seed=1')
    )
    calls, process = _process()
    run_one(context=ctx, extra_args=None, run_process=process)
    assert (study / 'submissions.jsonl').read_text() == before
    assert sorted(path.name for path in (study / 'jobs').glob('*')) == jobs_before
    assert len(calls) == 1

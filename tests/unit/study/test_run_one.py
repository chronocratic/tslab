"""``tslab study run``: one run in the foreground, nothing recorded."""

from collections.abc import Callable
import dataclasses
from pathlib import Path
import shlex

import pytest

from tests.unit.jobs.fake_slurm import FakeSlurm
from tests.unit.study.helpers import submit_study
from tslab.cli import main
from tslab.study.commands import run_one
from tslab.study.context import Context, load_context, Selection

GPUTWO_MACHINE = """\
launcher = "slurm"
output_root = "{output_root}"

[slurm]
partition = "CHANGE_ME"
time = "00:10:00"
gpus = 2
"""


def _one_run_context(*, project: Path, machine_file: str, only: tuple[str, ...]) -> Context:
    return load_context(
        reference='toy_studies:study',
        machine_path=project / machine_file,
        selection=Selection(only=only),
        project_root=Path.cwd(),
    )


def _process(*, code: int = 3) -> tuple[list[list[str]], Callable[[list[str]], int]]:
    calls: list[list[str]] = []

    def run_process(args: list[str]) -> int:
        calls.append(args)
        return code

    return calls, run_process


def _write_gpu_machine(*, project: Path) -> None:
    text = GPUTWO_MACHINE.format(output_root=(project / 'runs').as_posix())
    (project / 'gpus.toml').write_text(text)


def test_run_one_runs_the_command_of_one_run(project: Path, fake: FakeSlurm) -> None:
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


def test_run_one_reuses_extra_args_of_latest_submission(
    project: Path, fake: FakeSlurm, tmp_path: Path
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
    project: Path, fake: FakeSlurm, tmp_path: Path
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
    project: Path, fake: FakeSlurm, caplog: pytest.LogCaptureFixture
) -> None:
    submit_study(project=project)
    fake.set_answers('squeue', [{'exit': 0, 'stdout': '1000_0|RUNNING\n'}])
    ctx = _one_run_context(
        project=project, machine_file='cluster.toml', only=('model=small', 'seed=1')
    )
    calls, process = _process()
    with caplog.at_level('ERROR', logger='tslab.study.commands'):
        assert run_one(context=ctx, extra_args=None, run_process=process) == 1
    assert calls == []
    assert 'g/small/1 is running (job 1000_0); wait for it or cancel it first' in caplog.text


def test_run_one_without_a_selection_raises(project: Path, fake: FakeSlurm) -> None:
    submit_study(project=project)
    ctx = _one_run_context(project=project, machine_file='cluster.toml', only=('model=small',))
    empty = dataclasses.replace(ctx, reports=())
    with pytest.raises(ValueError, match='no runs match the selection'):
        run_one(context=empty, extra_args=None)


def test_run_one_cli_lists_more_than_one_run(
    project: Path, fake: FakeSlurm, caplog: pytest.LogCaptureFixture
) -> None:
    submit_study(project=project)
    with caplog.at_level('ERROR', logger='tslab.cli'), pytest.raises(SystemExit) as raised:
        main(args=['study', 'run', 'toy_studies:study', '--machine', 'cluster.toml'])
    assert raised.value.code == 2
    text = caplog.text
    assert '4 runs match the selection' in text
    assert 'g/small/1' in text
    assert 'g/large/2' in text
    assert 'narrow with --only' in text


def test_run_one_says_one_process_for_multi_gpu_jobs(
    project: Path, fake: FakeSlurm, caplog: pytest.LogCaptureFixture
) -> None:
    _write_gpu_machine(project=project)
    submit_study(project=project, machine_file='gpus.toml')
    ctx = _one_run_context(
        project=project, machine_file='gpus.toml', only=('model=small', 'seed=1')
    )
    calls, process = _process()
    with caplog.at_level('INFO', logger='tslab.study.commands'):
        run_one(context=ctx, extra_args=None, run_process=process)
    assert 'study run starts one process; the job would start 2' in caplog.text
    assert len(calls) == 1
    assert calls[0][:2] == ['bash', '-c']


def test_run_one_records_nothing(project: Path, fake: FakeSlurm) -> None:
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

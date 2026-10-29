"""Failure details and bounded log tails."""

import json
from pathlib import Path
from typing import cast

import pytest

from tests.unit.study.fake_slurm import FakeSlurm
from tests.unit.study.helpers import context, finish_run, study_directory, submit_study
from tslab.cli import main
from tslab.core.paths import ERROR_FILE, LOGS_DIRECTORY
from tslab.core.status import (
    InstanceRecord,
    InstanceState,
    RunState,
    RunStatusSnapshot,
    utc_now_iso,
    write_status,
)
from tslab.study.context import Context
from tslab.study.failures import collect_failures, log_failures
from tslab.study.result_table import instance_directory
from tslab.study.state import StudyState


def _failed_context(project: Path, fake: FakeSlurm) -> Context:
    submit_study(project=project)
    run_directory = study_directory(project=project) / 'g/small/1'
    write_status(
        path=run_directory / 'status.json',
        snapshot=RunStatusSnapshot(
            run_state=RunState.FINISHED,
            updated_at=utc_now_iso(),
            instances={
                'train[default]': InstanceRecord(
                    state=InstanceState.FAILED, error='ValueError: negative factor'
                )
            },
        ),
    )
    return context(project=project, machine_file='cluster.toml')


def test_failures_lists_error_and_log_tails(project: Path, fake: FakeSlurm) -> None:
    ctx = _failed_context(project, fake)
    run_directory = study_directory(project=project) / 'g/small/1'
    error_file = (
        instance_directory(run_directory=run_directory, instance_id='train[default]') / ERROR_FILE
    )
    error_file.parent.mkdir(parents=True)
    error_file.write_text('traceback details\n', encoding='utf-8')
    task = next(task for job in ctx.jobs for task in job.tasks if task.run_path == 'g/small/1')
    grid = ctx.study.grids[task.grid]
    job_log = ctx.study_directory / LOGS_DIRECTORY / f'{grid.short_name}_{task.index}.out'
    job_log.parent.mkdir(parents=True)
    job_log.write_text(''.join(f'line {index}\n' for index in range(100)), encoding='utf-8')
    run_log = run_directory / LOGS_DIRECTORY / 'run.log'
    run_log.parent.mkdir(parents=True)
    run_log.write_text('run started\nrun failed\n', encoding='utf-8')
    failures = collect_failures(context=ctx, states=(), lines=30)
    assert len(failures) == 1
    failure = failures[0]
    assert failure['state'] == StudyState.FAILED.value
    errors = cast('list[dict[str, object]]', failure['errors'])
    assert errors[0]['error_file'] == error_file.resolve().as_posix()
    assert errors[0]['error'] == 'ValueError: negative factor'
    assert failure['log_file'] == job_log.resolve().as_posix()
    assert len(cast('list[str]', failure['log_tail'])) == 30
    assert cast('list[str]', failure['log_tail'])[-1] == 'line 99'
    assert failure['run_log_tail'] == ['run started', 'run failed']


def test_failures_names_missing_files(
    project: Path, fake: FakeSlurm, caplog: pytest.LogCaptureFixture
) -> None:
    ctx = _failed_context(project, fake)
    failures = collect_failures(context=ctx, states=(), lines=30)
    failure = failures[0]
    assert failure['log_tail'] is None
    assert failure['run_log_tail'] is None
    with caplog.at_level('INFO', logger='tslab.study.failures'):
        log_failures(failures=failures)
    assert f'{failure["log_file"]}: missing' in caplog.text
    assert f'{failure["run_log"]}: missing' in caplog.text


def test_failures_default_states_skip_done_and_missing(project: Path, fake: FakeSlurm) -> None:
    ctx = _failed_context(project, fake)
    finish_run(project=project, model='large', seed=1)
    ctx = context(project=project, machine_file='cluster.toml')
    failures = collect_failures(context=ctx, states=(), lines=30)
    assert [failure['state'] for failure in failures] == [StudyState.FAILED.value]


def test_failures_json_shape(
    project: Path, fake: FakeSlurm, capsys: pytest.CaptureFixture[str]
) -> None:
    _failed_context(project, fake)
    with pytest.raises(SystemExit) as raised:
        main(args=['study', 'failures', 'toy_studies:study', '--machine', 'cluster.toml', '--json'])
    document = json.loads(capsys.readouterr().out)
    assert raised.value.code == 0
    assert document['schema'] == 'tslab.study-failures/1'
    assert len(document['runs']) == 1

"""Failure details, hints, and bounded log tails."""

import dataclasses
from datetime import timedelta
import json
from pathlib import Path, PurePosixPath
from typing import cast

import pytest

from tests.lab.unit.study.helpers import (
    context,
    finish_run,
    study_directory,
    submit_study,
    write_run_status,
)
from tests.support.fake_backend import FakeBackend
from tslab.lab.cli.entrypoint import main
from tslab.lab.core.paths import ERROR_FILE, LOGS_DIRECTORY
from tslab.lab.core.status import InstanceRecord, RunStatusSnapshot, utc_now_iso, write_status
from tslab.lab.enums import InstanceState, RunState, StudyState
from tslab.lab.study.context import Context
from tslab.lab.study.launchers.common.types import JobUsage
from tslab.lab.study.launchers.tables import backend_for
from tslab.lab.study.paths import instance_directory
from tslab.lab.study.reporting.failures import collect_failures, failure_hint, log_failures
from tslab.lab.study.reporting.state import RunReport


def _failed_context(project: Path) -> Context:
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


def test_failures_lists_error_and_log_tails(project: Path, fake_backend: FakeBackend) -> None:
    ctx = _failed_context(project=project)
    run_directory = study_directory(project=project) / 'g/small/1'
    error_file = (
        instance_directory(run_directory=run_directory, instance_id='train[default]') / ERROR_FILE
    )
    error_file.parent.mkdir(parents=True)
    error_file.write_text('traceback details\n', encoding='utf-8')
    task = next(
        task
        for job in ctx.jobs
        for task in job.tasks
        if task.run_path == PurePosixPath('g/small/1')
    )
    grid = ctx.study.grids[task.grid]
    logs = ctx.study_directory / '.study' / LOGS_DIRECTORY
    job_log = logs / f'{ctx.study.name}_{grid.short_name}_{task.index}.out'
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
    project: Path, fake_backend: FakeBackend, caplog: pytest.LogCaptureFixture
) -> None:
    ctx = _failed_context(project=project)
    failures = collect_failures(context=ctx, states=(), lines=30)
    failure = failures[0]
    assert failure['log_tail'] is None
    assert failure['run_log_tail'] is None
    with caplog.at_level('INFO', logger='tslab.lab.study.reporting.failures'):
        log_failures(failures=failures)
    assert f'{failure["log_file"]}: missing' in caplog.text
    assert f'{failure["run_log"]}: missing' in caplog.text


def test_failures_default_states_skip_done_and_missing(
    project: Path, fake_backend: FakeBackend
) -> None:
    _failed_context(project=project)
    finish_run(project=project, model='large', seed=1)
    ctx = context(project=project, machine_file='cluster.toml')
    ctx = dataclasses.replace(
        ctx,
        reports=[
            dataclasses.replace(report, state=StudyState.DEPENDENCY_FAILED)
            if report.run.run_path == PurePosixPath('g/small/2')
            else report
            for report in ctx.reports
        ],
    )
    failures = collect_failures(context=ctx, states=(), lines=30)
    assert [failure['state'] for failure in failures] == [
        StudyState.FAILED.value,
        StudyState.DEPENDENCY_FAILED.value,
    ]


def test_collect_failures_uses_selected_backend_usage(
    project: Path, fake_backend: FakeBackend
) -> None:
    ctx = _failed_context(project=project)
    collect_failures(context=ctx, states=(), lines=30)
    assert fake_backend.calls('failure_usage') == [{'context': ctx, 'reports': [ctx.reports[0]]}]


def test_failure_entry_shows_the_exit_code(
    project: Path, fake_backend: FakeBackend, caplog: pytest.LogCaptureFixture
) -> None:
    ctx = _failed_context(project=project)
    report = next(report for report in ctx.reports if report.state is StudyState.FAILED)
    assert report.job_id is not None
    fake_backend.failure_usage_answer = {
        report.job_id: JobUsage(
            state='FAILED',
            elapsed=timedelta(minutes=1),
            time_limit=timedelta(minutes=10),
            total_cpu=timedelta(seconds=30),
            max_rss=1024,
            alloc_cpus=2,
            gpus=0,
            exit_code='0:9',
        )
    }
    failures = collect_failures(context=ctx, states=(), lines=30)
    (failure,) = failures
    assert failure['exit_code'] == '0:9'
    with caplog.at_level('INFO', logger='tslab.lab.study.reporting.failures'):
        log_failures(failures=failures)
    assert 'exit 0:9' in caplog.text


def test_failures_without_sacct_still_lists_the_run(
    project: Path, fake_backend: FakeBackend
) -> None:
    ctx = _failed_context(project=project)
    # why: an empty answer is what the SLURM backend returns after a failed sacct call
    fake_backend.failure_usage_answer = {}
    failures = collect_failures(context=ctx, states=(), lines=30)
    assert len(failures) == 1
    assert failures[0]['state'] == StudyState.FAILED.value
    assert failures[0]['exit_code'] is None


def test_failures_json_shape(
    project: Path, fake_backend: FakeBackend, capsys: pytest.CaptureFixture[str]
) -> None:
    _failed_context(project=project)
    with pytest.raises(SystemExit) as raised:
        main(args=['study', 'failures', 'toy_studies:study', '--machine', 'cluster.toml', '--json'])
    document = json.loads(capsys.readouterr().out)
    assert raised.value.code == 0
    assert document['schema'] == 'tslab.study-failures'
    assert len(document['runs']) == 1


def _report(project: Path) -> RunReport:
    ctx = context(project=project, machine_file='cluster.toml')
    return ctx.reports[0]


def _hint(*, report: RunReport, project: Path, fake_backend: FakeBackend) -> str | None:
    ctx = context(project=project, machine_file='cluster.toml')
    return failure_hint(
        report=report,
        usage=None,
        backend=backend_for(machine=ctx.machine),
        resources=ctx.resolved[report.run.run_path].resources,
    )


def test_hint_for_signal_stop(project: Path, fake_backend: FakeBackend) -> None:
    report = dataclasses.replace(
        _report(project=project),
        reason='JobTimeLimitSignal: step job-0_0 received signal USR2',
        job_state='completed',
    )
    assert _hint(report=report, project=project, fake_backend=fake_backend) == (
        'stopped before the time limit; resubmit continues it, or raise time'
    )


def test_hint_for_paused_training(project: Path, fake_backend: FakeBackend) -> None:
    report = dataclasses.replace(
        _report(project=project),
        reason='TrainingPaused: job time share reached',
        job_state='completed',
    )
    assert _hint(report=report, project=project, fake_backend=fake_backend) == (
        'training used its share of the job (job_time_share); '
        'resubmit continues it, or raise time to train longer per job'
    )


def test_no_hint_for_python_error(project: Path, fake_backend: FakeBackend) -> None:
    report = dataclasses.replace(
        _report(project=project), reason='ValueError: bad', job_state='failed'
    )
    assert _hint(report=report, project=project, fake_backend=fake_backend) is None


def test_hint_comes_from_the_backend_after_the_reason_hints(
    project: Path, fake_backend: FakeBackend
) -> None:
    fake_backend.failure_hint_answer = 'backend hint'
    report = dataclasses.replace(_report(project=project), reason='ValueError: bad')
    assert _hint(report=report, project=project, fake_backend=fake_backend) == 'backend hint'
    paused = dataclasses.replace(report, reason='TrainingPaused: job time share reached')
    assert _hint(report=paused, project=project, fake_backend=fake_backend) == (
        'training used its share of the job (job_time_share); '
        'resubmit continues it, or raise time to train longer per job'
    )


def test_failures_json_has_hint(
    project: Path, fake_backend: FakeBackend, capsys: pytest.CaptureFixture[str]
) -> None:
    submit_study(project=project)
    write_run_status(
        project=project,
        model='small',
        seed=1,
        run_state=RunState.FAILED,
        instance=InstanceState.FAILED,
        error='TrainingPaused: job time share reached',
    )
    with pytest.raises(SystemExit) as raised:
        main(args=['study', 'failures', 'toy_studies:study', '--machine', 'cluster.toml', '--json'])
    document = json.loads(capsys.readouterr().out)
    assert raised.value.code == 0
    runs = {run['run_path']: run for run in cast('list[dict[str, object]]', document['runs'])}
    assert runs['g/small/1']['hint'] == (
        'training used its share of the job (job_time_share); '
        'resubmit continues it, or raise time to train longer per job'
    )
    assert 'g/small/2' not in runs


def test_failures_log_has_hint(
    project: Path, fake_backend: FakeBackend, caplog: pytest.LogCaptureFixture
) -> None:
    submit_study(project=project)
    write_run_status(
        project=project,
        model='small',
        seed=1,
        run_state=RunState.FAILED,
        instance=InstanceState.FAILED,
        error='TrainingPaused: job time share reached',
    )
    ctx = context(project=project, machine_file='cluster.toml')
    failures = collect_failures(context=ctx, states=(), lines=30)
    with caplog.at_level('INFO', logger='tslab.lab.study.reporting.failures'):
        log_failures(failures=failures)
    assert 'hint: training used its share of the job (job_time_share)' in caplog.text

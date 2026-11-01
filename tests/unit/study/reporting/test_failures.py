"""Failure details, hints, and bounded log tails."""

from collections.abc import Mapping, Sequence
import dataclasses
import json
from pathlib import Path
from typing import cast

import pytest

from tests.support.fake_slurm import FakeSlurm
from tests.unit.study.helpers import (
    context,
    finish_run,
    study_directory,
    submit_study,
    write_run_status,
)
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
from tslab.study import launchers
from tslab.study.context import Context
from tslab.study.launchers import JobUsage
from tslab.study.paths import instance_directory
import tslab.study.reporting.failures as failures_module
from tslab.study.reporting.failures import collect_failures, failure_hint, log_failures
from tslab.study.reporting.state import RunReport, StudyState


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
    with caplog.at_level('INFO', logger='tslab.study.reporting.failures'):
        log_failures(failures=failures)
    assert f'{failure["log_file"]}: missing' in caplog.text
    assert f'{failure["run_log"]}: missing' in caplog.text


def test_failures_default_states_skip_done_and_missing(project: Path, fake: FakeSlurm) -> None:
    ctx = _failed_context(project, fake)
    finish_run(project=project, model='large', seed=1)
    ctx = context(project=project, machine_file='cluster.toml')
    failures = collect_failures(context=ctx, states=(), lines=30)
    assert [failure['state'] for failure in failures] == [StudyState.FAILED.value]


def test_collect_failures_uses_selected_backend_usage(
    project: Path, fake: FakeSlurm, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _failed_context(project, fake)
    calls: list[tuple[Context, Sequence[RunReport]]] = []

    class Backend:
        def failure_usage(
            self, *, context: Context, reports: Sequence[RunReport]
        ) -> Mapping[str, JobUsage]:
            calls.append((context, reports))
            no_usage: dict[str, JobUsage] = {}
            return no_usage

        def failure_hint(self, **_kwargs: object) -> None:
            return None

    backend = Backend()
    monkeypatch.setattr(failures_module, 'backend_for', lambda **_kwargs: backend)

    collect_failures(context=ctx, states=(), lines=30)

    assert calls == [(ctx, [ctx.reports[0]])]


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


def _report(project: Path, fake: FakeSlurm) -> RunReport:
    ctx = context(project=project, machine_file='cluster.toml')
    return ctx.reports[0]


def _hint(*, report: RunReport, project: Path, fake: FakeSlurm) -> str | None:
    ctx = context(project=project, machine_file='cluster.toml')
    return failure_hint(
        report=report,
        usage=None,
        backend=launchers.backend_for(machine=ctx.machine),
        resources=ctx.resolved[report.run.run_path].resources,
    )


def test_hint_for_signal_stop(project: Path, fake: FakeSlurm) -> None:
    report = dataclasses.replace(
        _report(project, fake),
        reason='JobTimeLimitSignal: step 1000_0 received signal USR2',
        job_state='COMPLETED',
    )
    assert _hint(report=report, project=project, fake=fake) == (
        'stopped before the time limit; resubmit continues it, or raise time'
    )


def test_hint_for_paused_training(project: Path, fake: FakeSlurm) -> None:
    report = dataclasses.replace(
        _report(project, fake),
        reason='TrainingPaused: job time share reached',
        job_state='COMPLETED',
    )
    assert _hint(report=report, project=project, fake=fake) == (
        'training used its share of the job (job_time_share); '
        'resubmit continues it, or raise time to train longer per job'
    )


def test_no_hint_for_python_error(project: Path, fake: FakeSlurm) -> None:
    report = dataclasses.replace(
        _report(project, fake), reason='ValueError: bad', job_state='FAILED'
    )
    assert _hint(report=report, project=project, fake=fake) is None


def test_failures_json_has_hint(
    project: Path, fake: FakeSlurm, capsys: pytest.CaptureFixture[str]
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


def test_oom_hint_without_numbers_when_sacct_fails(
    project: Path, fake: FakeSlurm, caplog: pytest.LogCaptureFixture
) -> None:
    submit_study(project=project)
    fake.set_answers(
        'sacct',
        [
            {
                'exit': 0,
                'stdout': (
                    '1000_0|OUT_OF_MEMORY\n1000_1|COMPLETED\n1000_2|COMPLETED\n1000_3|COMPLETED\n'
                ),
            },
            {'exit': 1, 'stderr': 'sacct: error: down'},
        ],
    )
    ctx = context(project=project, machine_file='cluster.toml')
    with caplog.at_level('WARNING', logger='tslab.study.reporting.failures'):
        failures = collect_failures(context=ctx, states=(), lines=30)
    assert 'sacct' in caplog.text
    (failure,) = failures
    assert failure['state'] == StudyState.FAILED.value
    assert failure['hint'] == 'raise mem_per_cpu'


def test_failures_log_has_hint(
    project: Path, fake: FakeSlurm, caplog: pytest.LogCaptureFixture
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
    with caplog.at_level('INFO', logger='tslab.study.reporting.failures'):
        log_failures(failures=failures)
    assert 'hint: training used its share of the job (job_time_share)' in caplog.text

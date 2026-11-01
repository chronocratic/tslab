"""The behavior of the SLURM study backend."""

from dataclasses import replace
import json
import logging
from pathlib import Path
from typing import cast, TYPE_CHECKING

import pytest

from tests.support.fake_slurm import FakeSlurm
from tests.unit.study.helpers import context, jobs_of, no_sleep, submit_study
from tslab.cli import main
from tslab.jobs.slurm import (
    FAILING_JOB_STATES,
    INTERRUPTING_JOB_STATES,
    QUEUED_JOB_STATES,
    RUNNING_JOB_STATES,
)
from tslab.study.commands import cancel
from tslab.study.context import Context
from tslab.study.doctor import DoctorCheck
from tslab.study.launchers import JobPhase, JobUsage, LaunchError, LaunchReport
import tslab.study.launchers.slurm.backend as slurm_backend
from tslab.study.launchers.slurm.backend import _phase_of, SlurmBackend
from tslab.study.machine import MachineOverride, MachineProfileError
from tslab.study.records import JobRecord

if TYPE_CHECKING:
    from collections.abc import MutableMapping


def _context(*, project: Path, fake: FakeSlurm) -> Context:
    fake.set_answers('squeue', [])
    return context(project=project, machine_file='cluster.toml')


@pytest.mark.parametrize(
    ('job_state', 'phase'),
    [
        *((state, JobPhase.QUEUED) for state in sorted(QUEUED_JOB_STATES)),
        *((state, JobPhase.RUNNING) for state in sorted(RUNNING_JOB_STATES)),
        *((state, JobPhase.INTERRUPTED) for state in sorted(INTERRUPTING_JOB_STATES)),
        *((state, JobPhase.FAILED) for state in sorted(FAILING_JOB_STATES)),
        ('COMPLETED', JobPhase.COMPLETED),
        ('UNRECOGNIZED', JobPhase.ENDED),
        (None, JobPhase.UNKNOWN),
    ],
)
def test_phase_of_maps_scheduler_states(*, job_state: str | None, phase: JobPhase) -> None:
    assert _phase_of(job_state=job_state) is phase


def test_job_states_returns_phases_and_read_only_mappings(
    *, project: Path, fake: FakeSlurm
) -> None:
    submit_study(project=project)
    ctx = _context(project=project, fake=fake)
    fake.set_answers('squeue', [{'exit': 0, 'stdout': '1000_0|RUNNING\n1000_1|PENDING\n'}])
    fake.set_answers('sacct', [{'exit': 0, 'stdout': '1000_2|COMPLETED\n1000_3|COMPLETED\n'}])

    states, cache = SlurmBackend().job_states(
        machine=ctx.machine, study=ctx.study, jobs=jobs_of(project=project), cache={}
    )

    assert states.known
    assert states.states == {
        '1000_0': 'RUNNING',
        '1000_1': 'PENDING',
        '1000_2': 'COMPLETED',
        '1000_3': 'COMPLETED',
    }
    assert states.phases == {
        '1000_0': JobPhase.RUNNING,
        '1000_1': JobPhase.QUEUED,
        '1000_2': JobPhase.COMPLETED,
        '1000_3': JobPhase.COMPLETED,
    }
    assert states.followup_live_job_ids == ()
    assert cache == {'1000_2': 'COMPLETED', '1000_3': 'COMPLETED'}
    with pytest.raises(TypeError):
        cast('MutableMapping[str, str]', states.states)['1000_0'] = 'FAILED'
    with pytest.raises(TypeError):
        cast('MutableMapping[str, JobPhase]', states.phases)['1000_0'] = JobPhase.FAILED


def test_job_states_skips_cached_ended_jobs(*, project: Path, fake: FakeSlurm) -> None:
    submit_study(project=project)
    ctx = _context(project=project, fake=fake)
    fake.set_answers('squeue', [{'exit': 0, 'stdout': '1000_0|RUNNING\n1000_1|PENDING\n'}])
    cache = {'1000_2': 'COMPLETED', '1000_3': 'COMPLETED'}

    states, new_cache = SlurmBackend().job_states(
        machine=ctx.machine, study=ctx.study, jobs=jobs_of(project=project), cache=cache
    )

    assert states.states == {
        '1000_0': 'RUNNING',
        '1000_1': 'PENDING',
        '1000_2': 'COMPLETED',
        '1000_3': 'COMPLETED',
    }
    assert new_cache == cache
    assert '1000_0,1000_1' in ' '.join(fake.calls('squeue')[-1])


def test_job_states_returns_unknown_when_scheduler_is_unavailable(
    *, project: Path, fake: FakeSlurm, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _context(project=project, fake=fake)
    cache = {'ended': 'COMPLETED'}
    monkeypatch.setattr(slurm_backend, 'slurm_available', lambda: False)

    states, new_cache = SlurmBackend().job_states(
        machine=ctx.machine, study=ctx.study, jobs=(), cache=cache
    )

    assert not states.known
    assert states.states == {}
    assert states.phases == {}
    assert new_cache is cache


def test_job_states_returns_unknown_after_scheduler_error(
    *, project: Path, fake: FakeSlurm
) -> None:
    submit_study(project=project)
    ctx = _context(project=project, fake=fake)
    fake.set_answers('squeue', [{'exit': 1, 'stderr': 'squeue: cannot access scheduler'}])
    cache = {'ended': 'COMPLETED'}

    states, new_cache = SlurmBackend().job_states(
        machine=ctx.machine, study=ctx.study, jobs=jobs_of(project=project), cache=cache
    )

    assert not states.known
    assert new_cache is cache


def test_job_states_reports_live_followup_jobs(*, project: Path, fake: FakeSlurm) -> None:
    submit_study(project=project)
    ctx = _context(project=project, fake=fake)
    fake.set_answers('squeue', [{'exit': 0, 'stdout': '9000|PENDING\n'}])
    followup = JobRecord(
        kind='followup',
        submission_id='f1',
        time='2026-01-01T00:00:00+00:00',
        launcher='slurm',
        job_id='9000',
        job_file=2,
        tasks=(),
    )

    states, _ = SlurmBackend().job_states(
        machine=ctx.machine, study=ctx.study, jobs=[*jobs_of(project=project), followup], cache={}
    )

    assert states.followup_live_job_ids == ('9000',)
    assert states.phases['9000'] is JobPhase.QUEUED


def test_resources_applies_grid_overrides(*, project: Path, fake: FakeSlurm) -> None:
    ctx = _context(project=project, fake=fake)
    override = MachineOverride(match={'grid': ('g',)}, slurm={'cpus_per_task': 4})
    backend = SlurmBackend()

    base = backend.resources(
        machine=ctx.machine, overrides=(), grid_name='g', values={}, local_cpus=1
    )
    matched = backend.resources(
        machine=ctx.machine, overrides=(override,), grid_name='g', values={}, local_cpus=1
    )
    unmatched = backend.resources(
        machine=ctx.machine, overrides=(override,), grid_name='h', values={}, local_cpus=1
    )

    assert base.cpus == 1
    assert base.max_tasks_per_job == 1000
    assert base.describe() == 'partition=CHANGE_ME, time=00:10:00, cpus=1'
    assert matched.cpus == 4
    assert unmatched.cpus == 1


def test_resources_rejects_profile_without_slurm_table(*, project: Path, fake: FakeSlurm) -> None:
    ctx = _context(project=project, fake=fake)
    machine = replace(ctx.machine, slurm=None)

    with pytest.raises(MachineProfileError, match=r'a slurm launcher needs the \[slurm\] table'):
        SlurmBackend().resources(
            machine=machine, overrides=(), grid_name='g', values={}, local_cpus=1
        )


def test_cancellable_job_ids_returns_live_runs(*, project: Path, fake: FakeSlurm) -> None:
    submit_study(project=project)
    fake.set_answers('squeue', [{'exit': 0, 'stdout': '1000_0|RUNNING\n1000_1|PENDING\n'}])
    fake.set_answers('sacct', [{'exit': 0, 'stdout': '1000_2|COMPLETED\n1000_3|COMPLETED\n'}])
    ctx = context(project=project, machine_file='cluster.toml')

    assert SlurmBackend().cancellable_job_ids(reports=ctx.reports) == ['1000_0', '1000_1']


def test_detach_rejects_slurm_machine(*, project: Path, fake: FakeSlurm) -> None:
    ctx = _context(project=project, fake=fake)

    with pytest.raises(
        ValueError, match='--detach is for local machines; SLURM jobs already run in the background'
    ):
        SlurmBackend().detach(context=ctx, arguments=(), wait_command='tslab')


def test_doctor_checks_reports_scheduler_commands(*, fake: FakeSlurm) -> None:
    checks = SlurmBackend().doctor_checks()

    assert checks == (DoctorCheck(name='slurm', status='ok', message='SLURM commands found'),)
    assert fake.calls('sbatch') == []


def test_launch_delegates_to_shared_launch(
    *, project: Path, fake: FakeSlurm, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _context(project=project, fake=fake)
    expected = LaunchReport(submitted=(), not_submitted=(), error=None)
    monkeypatch.setattr(slurm_backend, 'launch', lambda **_kwargs: expected)

    report = SlurmBackend().launch(
        context=ctx,
        groups=(),
        folder=ctx.study_directory / '.study',
        dry_run=False,
        begin=None,
        sleep=no_sleep,
        task_stream=None,
        environment={},
        pid=123,
        host='test-host',
        now=lambda: '2026-01-02T03:04:05+00:00',
    )

    assert report is expected


def test_after_submit_delegates_to_followup_queue(
    *, project: Path, fake: FakeSlurm, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _context(project=project, fake=fake)
    monkeypatch.setattr(
        'tslab.study.launchers.slurm.followups.queue_followups', lambda **_kwargs: (7, ())
    )

    assert (
        SlurmBackend().after_submit(context=ctx, submitted=(), retry=lambda: 0, sleep=no_sleep) == 7
    )


def test_expected_starts_delegates_to_context(
    *, project: Path, fake: FakeSlurm, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _context(project=project, fake=fake)
    starts = {'1000_0': '2026-01-01T00:00:00'}
    monkeypatch.setattr(slurm_backend, 'expected_starts', lambda **_kwargs: starts)

    assert SlurmBackend().expected_starts(context=ctx) is starts


def test_usage_delegates_and_has_no_gpu_ids(
    *, project: Path, fake: FakeSlurm, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _context(project=project, fake=fake)
    usage: dict[str, JobUsage] = {}
    monkeypatch.setattr('tslab.study.launchers.slurm.usage.usage', lambda **_kwargs: usage)

    returned_usage, gpu_ids = SlurmBackend().usage(context=ctx)

    assert returned_usage is usage
    assert gpu_ids == {}


def test_failure_usage_delegates_to_failures(
    *, project: Path, fake: FakeSlurm, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _context(project=project, fake=fake)
    usage: dict[str, JobUsage] = {}
    monkeypatch.setattr('tslab.study.launchers.slurm.usage.failure_usage', lambda **_kwargs: usage)

    assert SlurmBackend().failure_usage(context=ctx, reports=()) is usage


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


def test_backend_cancel_raises_launch_error_on_scancel_failure(
    project: Path, fake: FakeSlurm
) -> None:
    ctx = _context_with_live_runs(project=project, fake=fake)
    fake.set_answers('scancel', [{'exit': 1, 'stderr': 'slurm_load_jobs error: Invalid job id'}])
    with pytest.raises(LaunchError, match='Invalid job id'):
        SlurmBackend().cancel(context=ctx, dry_run=False, signal=None)


def test_cancel_failure_logs_and_returns_error_code(
    project: Path, fake: FakeSlurm, caplog: pytest.LogCaptureFixture
) -> None:
    ctx = _context_with_live_runs(project=project, fake=fake)
    fake.set_answers('scancel', [{'exit': 1, 'stderr': 'slurm_load_jobs error: Invalid job id'}])
    with caplog.at_level(logging.ERROR):
        code, cancelled = cancel(context=ctx, dry_run=False)
    assert code == 1
    assert cancelled == []
    assert any(record.levelno == logging.ERROR for record in caplog.records)


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
    with caplog.at_level('INFO', logger='tslab.study.launchers.slurm.backend'):
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

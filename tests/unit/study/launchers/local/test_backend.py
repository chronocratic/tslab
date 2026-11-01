"""The behavior of the local study backend."""

from dataclasses import replace
from io import StringIO
import os
from pathlib import Path
from typing import cast, TYPE_CHECKING

import pytest

from tests.unit.study.helpers import (
    context,
    finished_pid,
    jobs_of,
    record_local_job,
    study_directory,
    write_launch,
)
from tslab.core.meta import hostname
from tslab.study.launchers import JobPhase, LaunchReport
import tslab.study.launchers.local.backend as local_backend
from tslab.study.launchers.local.backend import _phase_of, LocalBackend
from tslab.study.launchers.local.states import _is_local_job_id
from tslab.study.machine import FollowupSettings, LocalSettings
from tslab.study.plan import plan_jobs

if TYPE_CHECKING:
    from collections.abc import MutableMapping


@pytest.mark.parametrize(
    ('job_state', 'phase'),
    [
        ('pending', JobPhase.QUEUED),
        ('running', JobPhase.RUNNING),
        ('ended', JobPhase.ENDED),
        ('UNRECOGNIZED', JobPhase.ENDED),
        (None, JobPhase.UNKNOWN),
    ],
)
def test_phase_of_maps_local_states(*, job_state: str | None, phase: JobPhase) -> None:
    assert _phase_of(job_state=job_state) is phase


@pytest.mark.parametrize(
    ('job_id', 'expected'),
    [('local-123-4', True), ('1000', False), ('1000_4', False), ('worker-123-x', False)],
)
def test_local_job_id_recognizes_process_id_shape(*, job_id: str, expected: bool) -> None:
    assert _is_local_job_id(job_id=job_id) is expected


def test_job_states_reads_live_launch_and_returns_read_only_mappings(*, project: Path) -> None:
    ctx = context(project=project, machine_file='local.toml')
    task_ids = record_local_job(project=project)
    write_launch(
        folder=study_directory(project=project) / '.study',
        pid=os.getpid(),
        host=hostname(),
        tasks={task_ids[0]: 'running', task_ids[1]: 'pending'},
    )
    cache = {'unchanged': 'COMPLETED'}

    states, new_cache = LocalBackend().job_states(
        machine=ctx.machine, study=ctx.study, jobs=jobs_of(project=project), cache=cache
    )

    assert states.known
    assert states.states == {
        task_ids[0]: 'running',
        task_ids[1]: 'pending',
        task_ids[2]: 'ended',
        task_ids[3]: 'ended',
    }
    assert states.phases == {
        task_ids[0]: JobPhase.RUNNING,
        task_ids[1]: JobPhase.QUEUED,
        task_ids[2]: JobPhase.ENDED,
        task_ids[3]: JobPhase.ENDED,
    }
    assert states.followup_live_job_ids == ()
    assert new_cache is cache
    with pytest.raises(TypeError):
        cast('MutableMapping[str, str]', states.states)[task_ids[0]] = 'ended'
    with pytest.raises(TypeError):
        cast('MutableMapping[str, JobPhase]', states.phases)[task_ids[0]] = JobPhase.ENDED


def test_job_states_marks_tasks_ended_when_launch_is_not_live(*, project: Path) -> None:
    ctx = context(project=project, machine_file='local.toml')
    task_ids = record_local_job(project=project)
    write_launch(
        folder=study_directory(project=project) / '.study',
        pid=finished_pid(),
        host='other-host',
        tasks=dict.fromkeys(task_ids, 'running'),
    )

    states, _ = LocalBackend().job_states(
        machine=ctx.machine, study=ctx.study, jobs=jobs_of(project=project), cache={}
    )

    assert states.states == dict.fromkeys(task_ids, 'ended')
    assert states.phases == dict.fromkeys(task_ids, JobPhase.ENDED)


def test_job_states_marks_tasks_ended_without_launch_file(*, project: Path) -> None:
    ctx = context(project=project, machine_file='local.toml')
    task_ids = record_local_job(project=project)

    states, _ = LocalBackend().job_states(
        machine=ctx.machine, study=ctx.study, jobs=jobs_of(project=project), cache={}
    )

    assert states.states == dict.fromkeys(task_ids, 'ended')


def test_expected_starts_is_none(*, project: Path) -> None:
    ctx = context(project=project, machine_file='local.toml')

    assert LocalBackend().expected_starts(context=ctx) is None


def test_resources_uses_available_local_cpus(
    *, project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = context(project=project, machine_file='local.toml')
    monkeypatch.setattr(local_backend, '_usable_cpus', lambda: 7)

    resources = LocalBackend().resources(
        machine=ctx.machine, overrides=(), grid_name='g', values={}
    )

    assert resources.cpus == 7
    assert resources.max_tasks_per_job is None
    assert resources.describe() == 'local'


def test_cancellable_job_ids_is_empty() -> None:
    assert LocalBackend().cancellable_job_ids(reports=[]) == []


def test_after_submit_does_not_retry_without_auto_retry(*, project: Path) -> None:
    ctx = context(project=project, machine_file='local.toml')

    assert (
        LocalBackend().after_submit(
            context=ctx, submitted=(), retry=lambda: 3, sleep=lambda _seconds: None
        )
        == 0
    )


def test_after_submit_retries_submitted_jobs_when_auto_retry_is_enabled(*, project: Path) -> None:
    ctx = context(project=project, machine_file='local.toml')
    machine = replace(ctx.machine, followup=FollowupSettings(auto_retry=True))
    record_local_job(project=project)
    retry_calls: list[bool] = []

    def retry() -> int:
        retry_calls.append(True)
        return 3

    result = LocalBackend().after_submit(
        context=replace(ctx, machine=machine),
        submitted=jobs_of(project=project),
        retry=retry,
        sleep=lambda _seconds: None,
    )

    assert result == 3
    assert retry_calls == [True]


def test_after_submit_does_not_retry_empty_submission(*, project: Path) -> None:
    ctx = context(project=project, machine_file='local.toml')
    machine = replace(ctx.machine, followup=FollowupSettings(auto_retry=True))

    assert (
        LocalBackend().after_submit(
            context=replace(ctx, machine=machine),
            submitted=(),
            retry=lambda: 3,
            sleep=lambda _seconds: None,
        )
        == 0
    )


def test_detach_delegates_to_pid_file(*, project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = context(project=project, machine_file='local.toml')
    monkeypatch.setattr('tslab.study.launchers.local.pid_file.detach_launch', lambda **_kwargs: 7)

    assert (
        LocalBackend().detach(context=ctx, arguments=('--study',), wait_command='tslab status') == 7
    )


def test_usage_delegates_with_gpu_ids(*, project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = context(project=project, machine_file='local.toml')
    usage = {'local-1-1_0': object()}
    gpu_ids = {'local-1-1_0': '0'}
    monkeypatch.setattr(
        'tslab.study.launchers.local.usage.usage', lambda **_kwargs: (usage, gpu_ids)
    )

    returned_usage, returned_gpu_ids = LocalBackend().usage(context=ctx)

    assert returned_usage is usage
    assert returned_gpu_ids is gpu_ids


def test_failure_usage_and_doctor_checks_are_empty(*, project: Path) -> None:
    ctx = context(project=project, machine_file='local.toml')

    assert LocalBackend().failure_usage(context=ctx, reports=()) == {}
    assert LocalBackend().doctor_checks() == ()


def test_launch_delegates_to_shared_launch(
    *, project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = context(project=project, machine_file='local.toml')
    ctx = replace(ctx, machine=replace(ctx.machine, local=LocalSettings(keep_awake=False)))
    expected = LaunchReport(submitted=(), not_submitted=(), error=None)
    monkeypatch.setattr(local_backend, 'launch', lambda **_kwargs: expected)

    report = LocalBackend().launch(
        context=ctx,
        groups=(),
        folder=ctx.study_directory / '.study',
        dry_run=False,
        begin=None,
        sleep=lambda _seconds: None,
        task_stream=None,
        environment={},
    )

    assert report is expected


def test_launch_resolves_gpus_from_given_environment(*, project: Path) -> None:
    ctx = context(project=project, machine_file='local.toml')
    run = ctx.runs[0]
    machine = replace(ctx.machine, local=LocalSettings(gpus='auto'))
    ctx = replace(ctx, machine=machine)
    groups = plan_jobs(
        study=ctx.study,
        machine=machine,
        resolved=[ctx.resolved[run.run_path]],
        indices={run.run_path: 0},
        extra_args={},
        attempts={},
        first_job_file=1,
    )

    report = LocalBackend().launch(
        context=ctx,
        groups=groups,
        folder=ctx.study_directory / '.study',
        dry_run=False,
        begin=None,
        sleep=lambda _seconds: None,
        task_stream=StringIO(),
        environment={'CUDA_VISIBLE_DEVICES': '4'},
    )

    assert report is not None
    assert '| gpus 4' in groups[0].tasks[0].log_file.read_text()


def test_cancel_delegates_to_local_cancel(
    *, project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = context(project=project, machine_file='local.toml')
    result = (0, ['local-1-1'])
    monkeypatch.setattr('tslab.study.launchers.local.cancel.cancel_local', lambda **_kwargs: result)

    assert LocalBackend().cancel(context=ctx, dry_run=True, signal=None) is result

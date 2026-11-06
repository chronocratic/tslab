"""The behavior of the local study backend."""

from dataclasses import replace
import fcntl
from io import StringIO
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import threading
from typing import cast, TYPE_CHECKING

import pytest

from tests.unit.study.helpers import (
    context,
    finished_pid,
    jobs_of,
    no_sleep,
    record_local_job,
    study_directory,
    write_launch,
)
from tslab.core.meta import hostname
from tslab.enums import JobPhase, LocalTaskState, Retry
from tslab.jobs.local import GpuDiscovery
from tslab.study.code_snapshot import CodeSnapshot
from tslab.study.commands import submit
from tslab.study.commands.submit import submit_runs
from tslab.study.definition import StudyRun
from tslab.study.launchers.common.types import LaunchError, LaunchReport, Resources
import tslab.study.launchers.local.backend as local_backend
from tslab.study.launchers.local.backend import (
    _auto_gpu_count,
    _phase_of,
    LocalBackend,
    LocalResources,
)
from tslab.study.launchers.local.run import LocalLauncher
from tslab.study.launchers.local.states import _is_local_job_id
from tslab.study.machine import FollowupSettings, LocalSettings
from tslab.study.plan import JobGroup, plan_jobs, PlannedTask, ResolvedRun

if TYPE_CHECKING:
    from collections.abc import MutableMapping


@pytest.mark.parametrize(
    ('job_state', 'phase'),
    [
        (LocalTaskState.PENDING, JobPhase.QUEUED),
        (LocalTaskState.RUNNING, JobPhase.RUNNING),
        (LocalTaskState.ENDED, JobPhase.ENDED),
        (None, JobPhase.UNKNOWN),
    ],
)
def test_phase_of_maps_local_states(*, job_state: LocalTaskState | None, phase: JobPhase) -> None:
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
    assert states.reasons == {}
    assert states.flags == {}
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


def test_resources_uses_given_local_cpus(*, project: Path) -> None:
    ctx = context(project=project, machine_file='local.toml')

    resources = LocalBackend().resources(
        machine=ctx.machine, overrides=(), grid_name='g', values={}, local_cpus=3
    )

    assert resources.cpus == 3
    assert resources.max_tasks_per_job is None
    assert resources.describe() == 'local'
    assert resources.signal_problem() is None


def _local_resources(*, project: Path, local: LocalSettings, local_cpus: int) -> Resources:
    ctx = context(project=project, machine_file='local.toml')
    machine = replace(ctx.machine, local=local)
    return LocalBackend().resources(
        machine=machine, overrides=(), grid_name='g', values={}, local_cpus=local_cpus
    )


def test_cpus_per_run_splits_the_cpus_over_parallel_slots(*, project: Path) -> None:
    resources = _local_resources(project=project, local=LocalSettings(max_parallel=4), local_cpus=8)
    assert resources.cpus_per_run == 2


def test_cpus_per_run_with_one_slot_is_every_cpu(*, project: Path) -> None:
    resources = _local_resources(project=project, local=LocalSettings(max_parallel=1), local_cpus=8)
    assert resources.cpus_per_run == 8


def test_cpus_per_run_counts_gpu_lanes(*, project: Path) -> None:
    resources = _local_resources(
        project=project, local=LocalSettings(gpus=(0, 1), tasks_per_gpu=2), local_cpus=16
    )
    assert resources.cpus_per_run == 4


def test_cpus_per_run_never_drops_below_one(*, project: Path) -> None:
    resources = _local_resources(
        project=project, local=LocalSettings(max_parallel=16), local_cpus=4
    )
    assert resources.cpus_per_run == 1


def test_cpus_per_run_auto_without_nvidia_smi_uses_max_parallel(
    *, project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _auto_gpu_count.cache_clear()
    try:
        monkeypatch.setenv('PATH', str(tmp_path / 'no-binaries'))
        monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
        # why: no nvidia-smi and no Apple GPU, so the auto profile sees no GPU
        monkeypatch.setattr('tslab.jobs.local.gpus.platform.system', lambda: 'Linux')
        resources = _local_resources(
            project=project,
            local=LocalSettings(gpus=GpuDiscovery.AUTO, max_parallel=2),
            local_cpus=8,
        )
    finally:
        _auto_gpu_count.cache_clear()
    assert resources.cpus_per_run == 4


def test_submit_while_a_local_launch_runs_fails_at_once(
    *, project: Path, caplog: pytest.LogCaptureFixture
) -> None:
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    ctx = context(project=project, machine_file='local.toml')
    folder = ctx.study_directory / '.study'
    write_launch(folder=folder, pid=child.pid, host=hostname(), tasks={})
    lock = (folder / 'submit.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    codes: list[int] = []
    thread = threading.Thread(
        target=lambda: codes.append(
            submit(context=ctx, include_done=False, dry_run=False, extra_args=(), sleep=no_sleep)
        )
    )
    try:
        with caplog.at_level('ERROR'):
            thread.start()
            thread.join(timeout=10)
        assert not thread.is_alive()
        assert codes == [1]
        assert 'a local launch is running' in caplog.text
    finally:
        lock.close()
        if thread.is_alive():
            thread.join(timeout=10)
        if child.poll() is None:
            child.kill()
        child.wait()


def test_before_launch_does_not_skip_the_launch_check(
    *, project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = context(project=project, machine_file='local.toml')
    monkeypatch.setattr(
        LocalBackend, 'launch_blocker', lambda *_args, **_kwargs: 'a local launch is running'
    )
    before_launch_calls: list[bool] = []

    def before_launch() -> int:
        before_launch_calls.append(True)
        return 7

    code, submitted = submit_runs(
        context=ctx,
        reports=ctx.reports,
        dry_run=False,
        extra_args={},
        sleep=no_sleep,
        before_launch=before_launch,
    )
    assert code == 1
    assert submitted == []
    assert before_launch_calls == []


def test_cancellable_job_ids_is_empty() -> None:
    assert LocalBackend().cancellable_job_ids(reports=[]) == []


def test_after_submit_does_not_retry_without_retry(*, project: Path) -> None:
    ctx = context(project=project, machine_file='local.toml')

    assert (
        LocalBackend().after_submit(
            context=ctx, submitted=(), retry=lambda: 3, sleep=lambda _seconds: None
        )
        == 0
    )


def test_after_submit_retries_submitted_jobs_when_retry_is_enabled(*, project: Path) -> None:
    ctx = context(project=project, machine_file='local.toml')
    machine = replace(ctx.machine, followup=FollowupSettings(retry=Retry.RESUBMIT))
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
    machine = replace(ctx.machine, followup=FollowupSettings(retry=Retry.RESUBMIT))

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
    assert LocalBackend().doctor_checks(machine=ctx.machine) == ()


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
        start_delay_seconds=None,
        sleep=lambda _seconds: None,
        task_stream=None,
        environment={},
        pid=123,
        host='test-host',
        now=lambda: '2026-01-02T03:04:05+00:00',
    )

    assert report is expected


def test_launch_resolves_gpus_from_given_environment(*, project: Path) -> None:
    ctx = context(project=project, machine_file='local.toml')
    run = ctx.runs[0]
    machine = replace(ctx.machine, local=LocalSettings(gpus=GpuDiscovery.AUTO))
    ctx = replace(ctx, machine=machine)
    groups = plan_jobs(
        study=ctx.study,
        machine=machine,
        resolved=[ctx.resolved[run.run_path]],
        indices={run.run_path: 0},
        extra_args={},
        attempts={},
        first_job_file=1,
        environment={},
    )

    report = LocalBackend().launch(
        context=ctx,
        groups=groups,
        folder=ctx.study_directory / '.study',
        dry_run=False,
        start_delay_seconds=None,
        sleep=lambda _seconds: None,
        task_stream=StringIO(),
        environment={'CUDA_VISIBLE_DEVICES': '4'},
        pid=123,
        host='test-host',
        now=lambda: '2026-01-02T03:04:05+00:00',
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


def _code_snapshot_group(*, snapshot: CodeSnapshot, log_file: Path) -> JobGroup:
    task = PlannedTask(
        resolved=ResolvedRun(
            run=StudyRun(grid='g', values={}, run_path=PurePosixPath('p'), settings={}),
            settings={},
            resources=LocalResources(cpus=1, cpus_per_run=1),
        ),
        index=1,
        log_file=log_file,
        command=(
            sys.executable,
            '-c',
            "import os; print(os.environ['PYTHONPATH'], os.environ['TSLAB_CODE_SNAPSHOT_HASH'])",
        ),
        extra_args=(),
        attempt=1,
    )
    return JobGroup(
        grid='g',
        job_name='g',
        resources=LocalResources(cpus=1, cpus_per_run=1),
        job_file=1,
        tasks=(task,),
        code_snapshot=snapshot,
    )


def _launcher(tmp_path: Path) -> LocalLauncher:
    return LocalLauncher(
        project_root=tmp_path,
        env={},
        local=LocalSettings(),
        pid=123,
        host='test-host',
        now=lambda: '2026-01-02T03:04:05+00:00',
    )


def test_local_task_imports_from_code_snapshot_env(tmp_path: Path) -> None:
    folder = tmp_path / 'snapshots' / 'abc123'
    (folder / 'src').mkdir(parents=True)
    snapshot = CodeSnapshot(
        folder=folder, hash='abc123def456', commit='c0ffee00', dirty=False, untracked_copied=False
    )
    study_folder = tmp_path / 'study'
    log_file = study_folder / 'logs' / 'g1_1.out'
    log_file.parent.mkdir(parents=True)
    group = _code_snapshot_group(snapshot=snapshot, log_file=log_file)
    _launcher(tmp_path).submit(
        group=group, study_folder=study_folder, record=lambda **_kwargs: None, upstream={}
    )
    lines = [line for line in log_file.read_text().splitlines() if 'abc123def456' in line]
    assert len(lines) == 1
    assert lines[0].startswith(f'{folder}/src')


def test_local_missing_code_snapshot_stops_before_record(tmp_path: Path) -> None:
    folder = tmp_path / 'snapshots' / 'missing'
    folder.mkdir(parents=True)
    snapshot = CodeSnapshot(
        folder=folder, hash='deadbeef00', commit='c0ffee00', dirty=False, untracked_copied=False
    )
    study_folder = tmp_path / 'study'
    log_file = study_folder / 'logs' / 'g1_1.out'
    log_file.parent.mkdir(parents=True)
    group = _code_snapshot_group(snapshot=snapshot, log_file=log_file)
    recorded: list[str] = []

    def record(job_id: str) -> None:
        recorded.append(job_id)

    with pytest.raises(LaunchError) as excinfo:
        _launcher(tmp_path).submit(
            group=group, study_folder=study_folder, record=record, upstream={}
        )
    assert str(excinfo.value) == (
        f'code snapshot {folder}/src is missing; '
        'not running the live code; rerun with --new-code-snapshot'
    )
    assert recorded == []

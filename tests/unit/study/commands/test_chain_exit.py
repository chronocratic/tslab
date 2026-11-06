"""Exit status used by chained SLURM jobs after their run command ends."""

from dataclasses import replace
from importlib import import_module
from pathlib import Path, PurePosixPath
import time

import pytest

from tests.support.fake_backend import FakeBackend
from tests.unit.study.helpers import context
from tslab.core.status import InstanceRecord, RunStatusSnapshot, utc_now_iso, write_status
from tslab.enums import InstanceState, LauncherKind, RunState
from tslab.study.commands.chain_exit import chain_exit
from tslab.study.definition import StudyRun
from tslab.study.launchers.slurm.records import SlurmJobRecord
from tslab.study.paths import study_folder
from tslab.study.records import append_job, JobKind, ProgressSnapshot, TaskRecord
from tslab.study.reporting.state import FolderFacts

_STATE_MODULE = import_module('tslab.study.reporting.state')


def _job_record(*, run: StudyRun, job_file: int, index: int) -> SlurmJobRecord:
    task = TaskRecord(
        index=index,
        grid=run.grid,
        run_path=run.run_path,
        command=('python', 'run.py'),
        extra_args=(),
        resources={},
        attempt=1,
        progress=ProgressSnapshot(done_instances=0, newest_checkpoint_mtime=None),
    )
    return SlurmJobRecord(
        kind=JobKind.RUN,
        submission_id='s1',
        time=utc_now_iso(),
        launcher=LauncherKind.SLURM,
        job_id='41',
        job_file=job_file,
        tasks=(task,),
    )


@pytest.mark.parametrize(
    ('run_state', 'instance_state', 'expected'),
    [
        (RunState.FINISHED, InstanceState.DONE, 0),
        (RunState.FINISHED, InstanceState.FAILED, 1),
        (None, None, 1),
    ],
)
def test_chain_exit_is_zero_only_for_done(
    *,
    project: Path,
    fake_backend: FakeBackend,
    run_state: RunState | None,
    instance_state: InstanceState | None,
    expected: int,
) -> None:
    del fake_backend
    ctx = context(project=project, machine_file='cluster.toml')
    run = ctx.runs[0]
    job_file = 3
    index = 17
    job = _job_record(run=run, job_file=job_file, index=index)
    ctx = replace(ctx, jobs=[job])
    if run_state is not None and instance_state is not None:
        run_folder = ctx.study_directory / PurePosixPath(run.run_path)
        write_status(
            path=run_folder / 'status.json',
            snapshot=RunStatusSnapshot(
                run_state=run_state,
                updated_at=utc_now_iso(),
                instances={'train': InstanceRecord(state=instance_state)},
            ),
        )

    assert chain_exit(context=ctx, job_file=job_file, index=index) == expected


def test_chain_exit_rereads_records_when_context_snapshot_is_stale(
    project: Path, fake_backend: FakeBackend
) -> None:
    del fake_backend
    ctx = context(project=project, machine_file='cluster.toml')
    run = ctx.runs[0]
    job = _job_record(run=run, job_file=3, index=17)
    write_status(
        path=ctx.study_directory / run.run_path / 'status.json',
        snapshot=RunStatusSnapshot(
            run_state=RunState.FINISHED,
            updated_at=utc_now_iso(),
            instances={'train': InstanceRecord(state=InstanceState.DONE)},
        ),
    )
    append_job(
        path=study_folder(machine=ctx.machine, study=ctx.study) / 'submissions.jsonl', record=job
    )

    assert chain_exit(context=replace(ctx, jobs=[]), job_file=3, index=17) == 0


def test_chain_exit_waits_for_status_to_reach_shared_storage(
    project: Path, fake_backend: FakeBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    del fake_backend
    ctx = context(project=project, machine_file='cluster.toml')
    run = ctx.runs[0]
    job = _job_record(run=run, job_file=3, index=17)
    ctx = replace(ctx, jobs=[job])
    write_status(
        path=ctx.study_directory / run.run_path / 'status.json',
        snapshot=RunStatusSnapshot(
            run_state=RunState.FINISHED,
            updated_at=utc_now_iso(),
            instances={'train': InstanceRecord(state=InstanceState.DONE)},
        ),
    )
    read_folder = _STATE_MODULE.read_folder
    reads = 0

    def delayed_read(*, run_directory: Path) -> FolderFacts:
        nonlocal reads
        reads += 1
        if reads == 1:
            return FolderFacts(snapshot=None, read_error=None, lock_live=False)
        return read_folder(run_directory=run_directory)

    monkeypatch.setattr(_STATE_MODULE, 'read_folder', delayed_read)
    monkeypatch.setattr(time, 'sleep', lambda _seconds: None)

    assert chain_exit(context=ctx, job_file=3, index=17) == 0
    assert reads >= 2

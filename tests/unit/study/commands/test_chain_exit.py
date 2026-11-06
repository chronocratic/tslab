"""Exit status used by chained SLURM jobs after their run command ends."""

from dataclasses import replace
from pathlib import Path, PurePosixPath

import pytest

from tests.support.fake_backend import FakeBackend
from tests.unit.study.helpers import context
from tslab.core.status import InstanceRecord, RunStatusSnapshot, utc_now_iso, write_status
from tslab.enums import InstanceState, LauncherKind, RunState
from tslab.study.commands.chain_exit import chain_exit
from tslab.study.launchers.slurm.records import SlurmJobRecord
from tslab.study.paths import study_folder
from tslab.study.records import append_job, JobKind, ProgressSnapshot, TaskRecord


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
    job = SlurmJobRecord(
        kind=JobKind.RUN,
        submission_id='s1',
        time=utc_now_iso(),
        launcher=LauncherKind.SLURM,
        job_id='41',
        job_file=job_file,
        tasks=(task,),
    )
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
    task = TaskRecord(
        index=17,
        grid=run.grid,
        run_path=run.run_path,
        command=('python', 'run.py'),
        extra_args=(),
        resources={},
        attempt=1,
        progress=ProgressSnapshot(done_instances=0, newest_checkpoint_mtime=None),
    )
    job = SlurmJobRecord(
        kind=JobKind.RUN,
        submission_id='s1',
        time=utc_now_iso(),
        launcher=LauncherKind.SLURM,
        job_id='41',
        job_file=3,
        tasks=(task,),
    )
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

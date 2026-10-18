"""Follow-up jobs: the automatic retry and the W&B sync behind the jobs of the study."""

import dataclasses
from pathlib import Path
import sys

import pytest

from study.fake_slurm import FakeSlurm
from study.helpers import context, jobs_of, no_sleep, submit_study
from tslab.core.status import (
    InstanceRecord,
    InstanceState,
    RunState,
    RunStatusSnapshot,
    utc_now_iso,
    write_status,
)
from tslab.study.commands import resubmit, submit
from tslab.study.followups import queue_followups
from tslab.study.records import append_job, JobRecord, ProgressSnapshot, TaskRecord

# why: the shared toy project holds four runs; the auto-retry test needs six
STUDIES = '''
"""The studies of the toy project."""

import sys

import toy_script
from tslab.study import DEFAULT_RUN_SETTINGS, Grid, Study

study = Study(
    name='s',
    grids={
        'g': Grid(
            short_name='g1',
            command=(sys.executable, 'toy_script.py'),
            config=toy_script.ToyConfig,
            axes={
                'model': {
                    'small': {'model': 'small'},
                    'large': {'model': 'large'},
                },
                'seed': {'1': {'seed': 1}, '2': {'seed': 2}, '3': {'seed': 3}},
            },
            run_settings={**DEFAULT_RUN_SETTINGS, 'data': 'data'},
        ),
    },
)
'''


def with_followups(*, project: Path) -> None:
    """The [followup] table, on top of the written cluster profile."""
    profile = project / 'cluster.toml'
    text = profile.read_text() + '\n[followup]\nauto_retry = true\nwandb_sync = true\n'
    profile.write_text(text)


def job_file_of(*, project: Path, job_file: int, suffix: str) -> Path:
    return project / 'runs' / 's' / '.study' / 'jobs' / f'{job_file}.{suffix}'


def bump_attempts(*, project: Path, run_path: str, index: int, up_to: int) -> None:
    """More attempts of one run, recorded like real submissions."""
    for attempt in range(2, up_to + 1):
        record = JobRecord(
            kind='run',
            submission_id=f't{attempt}',
            time=utc_now_iso(),
            launcher='slurm',
            job_id=f'200{attempt}',
            job_file=attempt,
            tasks=(
                TaskRecord(
                    index=index,
                    grid='g',
                    run_path=run_path,
                    command=(sys.executable, 'toy_script.py'),
                    extra_args=(),
                    resources={},
                    attempt=attempt,
                    progress=ProgressSnapshot(done_instances=0, newest_checkpoint_mtime=None),
                ),
            ),
        )
        append_job(path=project / 'runs' / 's' / '.study' / 'submissions.jsonl', record=record)


@pytest.fixture
def project(project: Path) -> Path:
    folder = project
    (folder / 'toy_studies.py').write_text(STUDIES)
    return folder


def test_submit_queues_followups_after_the_arrays(project: Path, fake: FakeSlurm) -> None:
    with_followups(project=project)
    submit_study(project=project)
    assert len(fake.calls('sbatch')) == 3
    jobs = jobs_of(project=project)
    assert [job.kind for job in jobs] == ['run', 'followup', 'followup']
    retry, wbsync = jobs[1], jobs[2]
    assert not retry.tasks
    retry_script = job_file_of(
        project=project, job_file=retry.job_file, suffix='sbatch'
    ).read_text()
    assert '--job-name=retry' in retry_script
    assert '--dependency=afterany:1000' in retry_script
    assert '--array' not in retry_script
    assert '#SBATCH --time=00:30:00' in retry_script
    assert '--gres' not in retry_script
    retry_tasks = job_file_of(project=project, job_file=retry.job_file, suffix='tasks').read_text()
    machine = (project / 'cluster.toml').as_posix()
    assert retry_tasks == (
        f'0 1 uv run tslab study resubmit toy_studies:study --machine {machine} --auto\n'
    )
    wbsync_script = job_file_of(
        project=project, job_file=wbsync.job_file, suffix='sbatch'
    ).read_text()
    assert '--job-name=wbsync' in wbsync_script
    wbsync_tasks = job_file_of(
        project=project, job_file=wbsync.job_file, suffix='tasks'
    ).read_text()
    assert wbsync_tasks == f'0 1 uv run tslab wandb-sync {(project / "runs" / "s").as_posix()}\n'


def test_auto_retries_only_interrupted_runs_with_progress_under_the_cap(
    project: Path, fake: FakeSlurm, caplog: pytest.LogCaptureFixture
) -> None:
    submit_study(project=project)
    # a: one more done instance than its snapshot
    folder_a = project / 'runs' / 's' / 'g/small/1'
    folder_a.mkdir(parents=True)
    write_status(
        path=folder_a / 'status.json',
        snapshot=RunStatusSnapshot(
            run_state=RunState.RUNNING,
            updated_at=utc_now_iso(),
            instances={'train': InstanceRecord(state=InstanceState.DONE)},
        ),
    )
    # b: a checkpoint newer than its snapshot
    folder_b = project / 'runs' / 's' / 'g/small/2'
    folder_b.mkdir(parents=True)
    (folder_b / 'last.ckpt').write_text('ckpt')
    # d: progress would count, but it already used all three attempts
    folder_d = project / 'runs' / 's' / 'g/large/2'
    folder_d.mkdir(parents=True)
    write_status(
        path=folder_d / 'status.json',
        snapshot=RunStatusSnapshot(
            run_state=RunState.RUNNING,
            updated_at=utc_now_iso(),
            instances={'train': InstanceRecord(state=InstanceState.DONE)},
        ),
    )
    bump_attempts(project=project, run_path='g/large/2', index=4, up_to=3)
    fake.set_answers('squeue', [{'exit': 0, 'stdout': ''}])
    fake.set_answers(
        'sacct',
        [
            {
                'exit': 0,
                'stdout': (
                    '1000_0|TIMEOUT\n1000_1|TIMEOUT\n1000_2|TIMEOUT\n2003_4|TIMEOUT\n'
                    '1000_3|FAILED\n1000_5|COMPLETED\n'
                ),
            }
        ],
    )
    with caplog.at_level('INFO'):
        code = resubmit(
            context=context(project=project, machine_file='cluster.toml'), auto=True, sleep=no_sleep
        )
    assert code == 0
    retried = jobs_of(project=project)[-1]
    assert [t.run_path for t in retried.tasks] == ['g/small/1', 'g/small/2']
    assert all(t.attempt == 2 for t in retried.tasks)
    assert 'stuck: g/small/3 (attempt 1, no progress)' in caplog.text
    assert 'gave up: g/large/2 after 3 attempts' in caplog.text
    assert len(fake.calls('sbatch')) == 2


def test_followup_chain_stops_when_nothing_is_resubmitted(project: Path, fake: FakeSlurm) -> None:
    with_followups(project=project)
    submit_study(project=project)
    submitted = len(fake.calls('sbatch'))
    fake.set_answers('sacct', [{'exit': 0, 'stdout': ''}])
    code = resubmit(
        context=context(project=project, machine_file='cluster.toml'), auto=True, sleep=no_sleep
    )
    assert code == 0
    assert len(fake.calls('sbatch')) == submitted
    assert [job.kind for job in jobs_of(project=project)] == ['run', 'followup', 'followup']


@pytest.mark.slow
def test_no_followups_locally_or_on_dry_run(project: Path, fake: FakeSlurm) -> None:
    with_followups(project=project)
    code = submit(
        context=context(project=project, machine_file='cluster.toml'),
        include_done=False,
        dry_run=True,
        extra_args=(),
        sleep=no_sleep,
    )
    assert code == 0
    assert fake.calls('sbatch') == []
    assert [job.kind for job in jobs_of(project=project)] == []
    submit_study(project=project, machine_file='local.toml')
    assert fake.calls('sbatch') == []
    assert [job.kind for job in jobs_of(project=project)] == ['run']


def test_followup_sbatch_failure_exits_one_and_skips_the_rest(
    project: Path, fake: FakeSlurm, caplog: pytest.LogCaptureFixture
) -> None:
    with_followups(project=project)
    fake.set_answers(
        'sbatch',
        [{'exit': 0, 'stdout': '1000;tslab\n'}, {'exit': 1, 'stderr': 'sbatch: error: bad'}],
    )
    with caplog.at_level('ERROR'):
        code = submit(
            context=context(project=project, machine_file='cluster.toml'),
            include_done=False,
            dry_run=False,
            extra_args=(),
            sleep=no_sleep,
        )
    assert code == 1
    assert [job.kind for job in jobs_of(project=project)] == ['run']
    assert 'follow-up retry not submitted' in caplog.text
    assert len(fake.calls('sbatch')) == 2


def test_retry_followup_needs_a_study_reference(
    project: Path, fake: FakeSlurm, caplog: pytest.LogCaptureFixture
) -> None:
    with_followups(project=project)
    ctx = dataclasses.replace(context(project=project, machine_file='cluster.toml'), reference='')
    with caplog.at_level('WARNING'):
        code, records = queue_followups(context=ctx, job_ids=['1000'], sleep=no_sleep)
    assert code == 0
    assert len(records) == 1
    assert 'automatic retry follow-up is skipped' in caplog.text

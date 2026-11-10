"""The ``slurm.sh`` file of a study: its commands, its bash and its write after a submit."""

# ruff: noqa: S603, S607 # why: the tests run bash and the fake SLURM tools

import dataclasses
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from tests.lab.unit.study.configs import grid
from tests.lab.unit.study.helpers import context, jobs_of, no_sleep, study_directory, submit_study
from tests.support.fake_slurm import FakeSlurm
from tslab.lab.core.status import utc_now_iso
from tslab.lab.enums import LauncherKind
from tslab.lab.jobs.slurm import CommandValues, SLURM_COMMANDS
from tslab.lab.study import Study
from tslab.lab.study.commands import submit
from tslab.lab.study.launchers.slurm import backend as slurm_backend
from tslab.lab.study.launchers.slurm.backend import SlurmBackend
from tslab.lab.study.launchers.slurm.records import SlurmJobRecord
from tslab.lab.study.launchers.slurm.slurm_file import (
    command_values,
    SLURM_FILE,
    slurm_file_text,
    write_slurm_file,
)
from tslab.lab.study.records import append_job, JobKind, submission_in_progress

STUDY = Study(name='aug', grids={'a': grid(short_name='a', job_name='{short_name}_{seed}')})
VALUES = CommandValues(job_ids=('1000', '1001'), width=8, start='2026-11-01')


def _record(
    *, job_id: str, time: str, kind: JobKind = JobKind.RUN, job_name: str | None = None
) -> SlurmJobRecord:
    return SlurmJobRecord(
        kind=kind,
        submission_id='s1',
        time=time,
        launcher=LauncherKind.SLURM,
        job_id=job_id,
        job_file=1,
        tasks=(),
        job_name=job_name,
    )


def test_slurm_file_defines_every_command_and_no_tool_name() -> None:
    text = slurm_file_text(study=STUDY, values=VALUES)
    for command in SLURM_COMMANDS:
        assert f'# {command.name}: {command.description}' in text
        assert f'\n{command.name}() {{' in text
    assert not re.search(r'^(squeue|sacct|seff|sstat|sprio|scontrol)\(\)', text, re.MULTILINE)


def test_slurm_file_is_valid_bash(tmp_path: Path) -> None:
    path = tmp_path / SLURM_FILE
    path.write_text(slurm_file_text(study=STUDY, values=VALUES))
    result = subprocess.run(['bash', '-n', str(path)], check=False)
    assert result.returncode == 0


def _run_bash(*, path: Path, call: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ['bash', '-c', f'source {path} && {call}'], capture_output=True, text=True, check=False
    )


def test_slurm_file_squeue_with_no_live_jobs(tmp_path: Path, fake: FakeSlurm) -> None:
    fake.set_answers('sacct', [{'stdout': '1000|COMPLETED\n1001|COMPLETED\n'}])
    path = tmp_path / SLURM_FILE
    path.write_text(slurm_file_text(study=STUDY, values=VALUES))
    result = _run_bash(path=path, call='squeue-full-name')
    assert (result.returncode, result.stdout) == (0, '')
    assert fake.calls('squeue') == []


def test_slurm_file_squeue_gets_live_ids_and_width(tmp_path: Path, fake: FakeSlurm) -> None:
    fake.set_answers('sacct', [{'stdout': '1000|COMPLETED\n1001_2|RUNNING\n'}])
    fake.set_answers('squeue', [{'stdout': 'row\n'}])
    path = tmp_path / SLURM_FILE
    path.write_text(slurm_file_text(study=STUDY, values=VALUES))
    result = _run_bash(path=path, call='squeue-full-name')
    assert (result.returncode, result.stdout) == (0, 'row\n')
    [call] = fake.calls('squeue')
    assert call[:3] == ['--array', '--jobs', '1001']
    assert '%.8j' in call[3]


def test_slurm_file_squeue_falls_back_and_tolerates_purged_ids(
    tmp_path: Path, fake: FakeSlurm
) -> None:
    fake.set_answers('sacct', [{'exit': 1}])
    fake.set_answers('squeue', [{'exit': 1, 'stderr': 'Invalid job id specified'}])
    path = tmp_path / SLURM_FILE
    path.write_text(slurm_file_text(study=STUDY, values=VALUES))
    assert _run_bash(path=path, call='squeue-full-name').returncode == 0
    assert fake.calls('squeue')[0][:3] == ['--array', '--jobs', '1000,1001']


def test_slurm_file_per_job_command_runs_once_per_id(tmp_path: Path, fake: FakeSlurm) -> None:
    path = tmp_path / SLURM_FILE
    path.write_text(slurm_file_text(study=STUDY, values=VALUES))
    assert _run_bash(path=path, call='seff-all').returncode == 0
    assert fake.calls('seff') == [['1000'], ['1001']]


def test_command_values_width_and_start(tmp_path: Path) -> None:
    study = Study(
        name='aug', grids={'a': grid(short_name='a_long_name_of_19', job_name='{short_name}')}
    )
    folder = tmp_path / '.study'
    append_job(
        path=folder / 'submissions.jsonl',
        record=_record(job_id='1000', time='2026-11-02T10:00:00+00:00'),
    )
    append_job(
        path=folder / 'submissions.jsonl',
        record=_record(job_id='1001', time='2026-11-03T10:00:00+00:00'),
    )
    append_job(
        path=folder / 'submissions.jsonl',
        record=_record(job_id='1000', time='2026-11-03T11:00:00+00:00'),
    )
    assert command_values(study=study, study_folder=folder) == CommandValues(
        job_ids=('1000', '1001'), width=len('a_long_name_of_19'), start='2026-11-01'
    )


def test_command_values_width_follows_recorded_job_names(tmp_path: Path) -> None:
    # why: a job from a submit with an older, longer template must not be cut
    folder = tmp_path / '.study'
    append_job(
        path=folder / 'submissions.jsonl',
        record=_record(
            job_id='1000', time='2026-11-02T10:00:00+00:00', job_name='an_older_and_longer_job_name'
        ),
    )
    append_job(
        path=folder / 'submissions.jsonl',
        record=_record(job_id='1001', time='2026-11-02T10:00:00+00:00'),
    )
    values = command_values(study=STUDY, study_folder=folder)
    assert values.width == len('an_older_and_longer_job_name')


def test_submit_writes_slurm_file(project: Path, fake: FakeSlurm) -> None:
    submit_study(project=project)
    text = (study_directory(project=project) / '.study' / SLURM_FILE).read_text()
    assert "_tslab_job_ids='1000'" in text
    assert jobs_of(project=project)


def test_dry_run_writes_no_slurm_file(project: Path, fake: FakeSlurm) -> None:
    code = submit(
        context=context(project=project, machine_file='cluster.toml'),
        include_done=False,
        dry_run=True,
        extra_args=(),
        sleep=no_sleep,
    )
    assert code == 0
    assert not (study_directory(project=project) / '.study' / SLURM_FILE).exists()


def test_write_slurm_file_returns_the_path(tmp_path: Path) -> None:
    folder = tmp_path / '.study'
    append_job(path=folder / 'submissions.jsonl', record=_record(job_id='7', time=utc_now_iso()))
    assert write_slurm_file(study=STUDY, study_folder=folder) == folder / SLURM_FILE


def test_submit_writes_catalogue_while_holding_submission_lock(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        'tslab.lab.study.launchers.slurm.followups.queue_followups', lambda **_kwargs: (0, ())
    )

    def check_lock(*, study: Study, study_folder: Path) -> Path:
        assert submission_in_progress(study_folder=study_folder)
        return study_folder / SLURM_FILE

    monkeypatch.setattr(slurm_backend, 'write_slurm_file', check_lock)
    code = SlurmBackend().after_submit(
        context=context(project=project, machine_file='cluster.toml'),
        submitted=(_record(job_id='1000', time=utc_now_iso()),),
        retry=lambda: 0,
        sleep=no_sleep,
    )

    assert code == 0


def test_command_values_skip_local_job_ids(tmp_path: Path) -> None:
    folder = tmp_path / '.study'
    path = folder / 'submissions.jsonl'
    append_job(path=path, record=_record(job_id='1000', time='2026-11-02T10:00:00+00:00'))
    local = _record(job_id='host-123-4', time='2026-11-02T11:00:00+00:00')
    append_job(path=path, record=dataclasses.replace(local, launcher=LauncherKind.LOCAL))
    assert command_values(study=STUDY, study_folder=folder).job_ids == ('1000',)


def test_slurm_file_per_job_exit_code_is_the_worst_one(tmp_path: Path, fake: FakeSlurm) -> None:
    fake.set_answers('seff', [{'exit': 1, 'stderr': 'bad'}, {'exit': 0}])
    path = tmp_path / SLURM_FILE
    path.write_text(slurm_file_text(study=STUDY, values=VALUES))
    assert _run_bash(path=path, call='seff-all').returncode == 1


def test_slurm_file_fallback_tolerates_purged_ids_for_every_live_tool(
    tmp_path: Path, fake: FakeSlurm
) -> None:
    fake.set_answers('sacct', [{'exit': 1}])
    fake.set_answers('scontrol', [{'exit': 1, 'stderr': 'Invalid job id specified'}])
    path = tmp_path / SLURM_FILE
    path.write_text(slurm_file_text(study=STUDY, values=VALUES))
    assert _run_bash(path=path, call='scontrol-show').returncode == 0


@pytest.mark.skipif(shutil.which('zsh') is None, reason='needs zsh')
def test_slurm_file_per_job_command_works_in_zsh(tmp_path: Path, fake: FakeSlurm) -> None:
    path = tmp_path / SLURM_FILE
    path.write_text(slurm_file_text(study=STUDY, values=VALUES))
    result = subprocess.run(
        ['zsh', '-c', f'source {path} && seff-all'], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0
    assert fake.calls('seff') == [['1000'], ['1001']]


def test_slurm_file_fallback_says_that_sacct_did_not_answer(
    tmp_path: Path, fake: FakeSlurm
) -> None:
    fake.set_answers('sacct', [{'exit': 1}])
    path = tmp_path / SLURM_FILE
    path.write_text(slurm_file_text(study=STUDY, values=VALUES))
    result = _run_bash(path=path, call='squeue-stock')
    assert 'sacct did not answer; asking every recorded job id' in result.stderr

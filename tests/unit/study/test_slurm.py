"""SLURM job scripts, sbatch backoff, and job state queries."""

import os
from pathlib import Path
import shlex
import subprocess

import pytest

from tests.unit.study.fake_slurm import FakeSlurm, install_fake_slurm
from tslab.study.machine import SlurmSettings
from tslab.study.slurm import (
    _as_sbatch_flag,
    array_spec,
    job_script,
    query_job_states,
    sbatch,
    SlurmError,
)

TASKS_LINES = [
    '3 1 ' + shlex.join(['python3', '-c', 'print(3)']),
    # why: the arguments hold a space and an apostrophe, the quoting round trip
    '17 2 ' + shlex.join(['python3', '-c', 'import sys; print(sys.argv[1:])', 'a b', "it's"]),
]


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeSlurm:
    return install_fake_slurm(tmp_path=tmp_path, monkeypatch=monkeypatch, answers={})


def write_script(tmp_path: Path, tasks_file: Path) -> Path:
    tasks_file.write_text('\n'.join(TASKS_LINES) + '\n')
    script = tmp_path / 'job.sbatch'
    script.write_text(
        job_script(
            job_name='cls',
            slurm=SlurmSettings(partition='CHANGE_ME', time='1:00:00', node_local_tmp=False),
            indices=(3, 17),
            tasks_file=tasks_file,
            log_pattern=tmp_path / 'logs' / 'cls_%a.out',
            project_root=tmp_path,
            env={},
        )
    )
    return script


def run_task_section(script: Path, task_id: str) -> subprocess.CompletedProcess[str]:
    body = [line for line in script.read_text().splitlines() if not line.startswith('#')]
    return subprocess.run(  # noqa: S603 # why: run the generated script under bash
        ['bash', '-c', '\n'.join(body)],  # noqa: S607 # why: bash is a PATH command
        check=False,
        env={
            **os.environ,
            'SLURM_ARRAY_TASK_ID': task_id,
            'SLURM_ARRAY_JOB_ID': '9',
            'SLURM_JOB_ID': '9',
        },
        capture_output=True,
        text=True,
    )


def test_job_script_header_and_task(tmp_path: Path) -> None:
    slurm = SlurmSettings(
        partition='CHANGE_ME',
        time='1:00:00',
        qos='normal',
        gpus=1,
        sbatch_extra=('--exclude=n1',),
        node_local_tmp=False,
    )
    text = job_script(
        job_name='cls',
        slurm=slurm,
        indices=(0, 1, 2, 3, 7, 9, 10),
        tasks_file=tmp_path / '3.tasks',
        log_pattern=tmp_path / 'logs' / 'cls_%a.out',
        project_root=Path('/p q'),
        env={'A': 'x y'},
    )
    lines = text.splitlines()
    assert '#SBATCH --array=0-3,7,9-10%100' in lines
    assert '#SBATCH --gres=gpu:1' in lines
    assert '#SBATCH --job-name=cls' in lines
    assert '#SBATCH --open-mode=append' in lines
    assert '#SBATCH --exclude=n1' in lines
    assert '#SBATCH --no-requeue' in lines
    assert not any('mem-per-cpu' in line for line in lines)
    assert "cd '/p q'" in lines
    assert "export A='x y'" in lines


def test_task_line_round_trips_through_bash(tmp_path: Path) -> None:
    script = write_script(tmp_path, tmp_path / '3.tasks')
    result = run_task_section(script, task_id='17')
    assert result.returncode == 0
    assert result.stdout.splitlines()[0].startswith('=== tslab attempt 2 | job 9_17 | ')
    assert result.stdout.splitlines()[1] == "['a b', \"it's\"]"


def test_followup_task_line_round_trips_through_bash(tmp_path: Path) -> None:
    # why: the follow-up jobs use the non-array section, whose tasks file
    # lines also start with the index and attempt
    tasks_file = tmp_path / 'followup.tasks'
    tasks_file.write_text('0 1 ' + shlex.join(['python3', '-c', "print('followed')"]) + '\n')
    script = tmp_path / 'followup.sbatch'
    script.write_text(
        job_script(
            job_name='wbsync',
            slurm=SlurmSettings(partition='CHANGE_ME', time='00:10:00', node_local_tmp=False),
            indices=None,
            tasks_file=tasks_file,
            log_pattern=tmp_path / 'logs' / 'wbsync.out',
            project_root=tmp_path,
            env={},
        )
    )
    result = run_task_section(script, task_id='0')
    assert result.returncode == 0
    assert result.stdout.splitlines()[-1] == 'followed'


def test_missing_task_index_fails_with_a_message(tmp_path: Path) -> None:
    tasks_file = tmp_path / '3.tasks'
    script = write_script(tmp_path, tasks_file)
    result = run_task_section(script, task_id='5')
    assert result.returncode == 1
    assert '5' in result.stderr
    assert str(tasks_file) in result.stderr


def test_array_spec() -> None:
    assert array_spec(indices=(10, 9, 0, 1, 2, 3, 7, 3)) == '0-3,7,9-10'
    assert array_spec(indices=(4,)) == '4'
    with pytest.raises(ValueError, match='indices'):
        array_spec(indices=())


def test_sbatch_backs_off_on_qos_limits(fake: FakeSlurm, tmp_path: Path) -> None:
    fake.set_answers(
        'sbatch',
        [{'exit': 1, 'stderr': 'error: QOSMaxSubmitJobPerUserLimit'}] * 7
        + [{'exit': 0, 'stdout': '555;cluster'}],
    )
    delays: list[float] = []
    assert sbatch(script=tmp_path / 'x.sbatch', sleep=delays.append) == '555'
    assert delays == [10, 20, 40, 80, 160, 320, 640]


def test_sbatch_backoff_caps_at_900(fake: FakeSlurm, tmp_path: Path) -> None:
    fake.set_answers(
        'sbatch',
        [{'exit': 1, 'stderr': 'error: QOSMaxSubmitJobPerUserLimit'}] * 9
        + [{'exit': 0, 'stdout': '556;cluster'}],
    )
    delays: list[float] = []
    assert sbatch(script=tmp_path / 'x.sbatch', sleep=delays.append) == '556'
    assert delays[-2:] == [900, 900]


def test_sbatch_other_errors_raise(fake: FakeSlurm, tmp_path: Path) -> None:
    fake.set_answers('sbatch', [{'exit': 1, 'stderr': 'error: invalid partition'}])
    with pytest.raises(SlurmError, match='invalid partition'):
        sbatch(script=tmp_path / 'x.sbatch', sleep=lambda _: None)


def test_query_job_states(fake: FakeSlurm) -> None:
    fake.set_answers('squeue', [{'exit': 0, 'stdout': '7_0|RUNNING\n7_1|PENDING\n'}])
    fake.set_answers(
        'sacct', [{'exit': 0, 'stdout': '7_0|RUNNING\n6_0|CANCELLED by 12\n6_1|TIMEOUT\nbad\n'}]
    )
    assert query_job_states(job_ids=['6', '7']) == {
        '7_0': 'RUNNING',
        '7_1': 'PENDING',
        '6_0': 'CANCELLED',
        '6_1': 'TIMEOUT',
    }
    assert fake.calls('squeue')[0][:2] == ['--array', '--jobs']


def test_query_job_states_raises_when_squeue_fails(fake: FakeSlurm) -> None:
    fake.set_answers('squeue', [{'exit': 1, 'stderr': 'slurm_load_jobs error: Socket timed out'}])
    with pytest.raises(SlurmError, match='Socket timed out'):
        query_job_states(job_ids=['7'])


def test_query_job_states_raises_when_sacct_fails(fake: FakeSlurm) -> None:
    fake.set_answers('squeue', [{'exit': 0, 'stdout': '7_0|RUNNING\n'}])
    fake.set_answers('sacct', [{'exit': 1, 'stderr': 'sacct: error: Problem talking to slurmdbd'}])
    with pytest.raises(SlurmError, match='slurmdbd'):
        query_job_states(job_ids=['7'])


def test_query_job_states_accepts_squeue_not_knowing_a_job(fake: FakeSlurm) -> None:
    fake.set_answers(
        'squeue', [{'exit': 1, 'stderr': 'slurm_load_jobs error: Invalid job id specified'}]
    )
    fake.set_answers('sacct', [{'exit': 0, 'stdout': '6_0|COMPLETED\n'}])
    assert query_job_states(job_ids=['6']) == {'6_0': 'COMPLETED'}


def _script(tmp_path: Path, **changes: object) -> str:
    values: dict[str, object] = {
        'job_name': 'cls',
        'slurm': SlurmSettings(partition='p', time='1:00:00'),
        'indices': (0,),
        'tasks_file': tmp_path / '1.tasks',
        'log_pattern': tmp_path / 'logs' / 'cls_%a.out',
        'project_root': tmp_path,
        'env': {},
    }
    return job_script(**{**values, **changes})  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    'changes',
    [
        {'job_name': 'a\nrm -rf x'},
        {'dependency': 'afterok:1\nrm -rf x'},
        {'begin': 'now\nrm'},
        {'slurm': SlurmSettings(partition='p\nrm', time='1:00:00')},
        {'slurm': SlurmSettings(partition='p', time='1:00:00', sbatch_extra=('--x\nrm',))},
    ],
)
def test_job_script_rejects_line_breaks(tmp_path: Path, changes: dict[str, object]) -> None:
    with pytest.raises(ValueError, match='line break'):
        _script(tmp_path, **changes)


@pytest.mark.parametrize('name', ['A;rm -rf x', '1X', 'A B', 'A=B'])
def test_job_script_rejects_bad_env_names(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match='environment variable name'):
        _script(tmp_path, env={name: 'v'})


def test_missing_task_message_survives_a_hostile_path(tmp_path: Path) -> None:
    folder = tmp_path / 'a"; touch pwned; "b'
    folder.mkdir()
    tasks_file = folder / '3.tasks'
    script = write_script(tmp_path, tasks_file)
    result = run_task_section(script, task_id='5')
    assert result.returncode == 1
    assert str(tasks_file) in result.stderr
    assert not (tmp_path / 'pwned').exists()
    assert not Path('pwned').exists()


def test_as_sbatch_flag() -> None:
    assert _as_sbatch_flag(key='cpus_per_task', value=4) == '#SBATCH --cpus-per-task=4'
    assert _as_sbatch_flag(key='exclusive', value=True) == '#SBATCH --exclusive'
    value = '/a b/x_%a.out'
    assert _as_sbatch_flag(key='output', value=value) == "#SBATCH --output='/a b/x_%a.out'"


def test_job_script_quotes_values_with_spaces(tmp_path: Path) -> None:
    script = _script(tmp_path, job_name='my job', log_pattern=tmp_path / 'a b' / 'x_%a.out')
    assert "#SBATCH --job-name='my job'" in script
    assert f"#SBATCH --output='{tmp_path}/a b/x_%a.out'" in script


def test_job_script_exports_pythonunbuffered_before_env(tmp_path: Path) -> None:
    lines = _script(tmp_path, env={'PYTHONUNBUFFERED': '0', 'A': '1'}).splitlines()
    assert 'export PYTHONUNBUFFERED=1' in lines
    # a value from [env] still wins
    assert lines.index('export PYTHONUNBUFFERED=1') < lines.index('export PYTHONUNBUFFERED=0')

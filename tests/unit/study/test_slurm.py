"""SLURM job scripts, sbatch backoff, and job state queries."""

import os
from pathlib import Path
import shlex
import subprocess
import time

import pytest

from tests.unit.study.fake_slurm import FakeSlurm, install_fake_slurm
from tslab.study.machine import SlurmSettings
from tslab.study.slurm import (
    _as_sbatch_flag,
    _expand_array_key,
    _sbatch_env,
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


def write_script(tmp_path: Path, tasks_file: Path, *, write_tasks: bool = True) -> Path:
    if write_tasks:
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


def _task_section_command(script: Path) -> list[str]:
    body = [line for line in script.read_text().splitlines() if not line.startswith('#')]
    return ['bash', '-c', '\n'.join(body)]


def _task_section_env(task_id: str) -> dict[str, str]:
    return {
        **os.environ,
        'SLURM_ARRAY_TASK_ID': task_id,
        'SLURM_ARRAY_JOB_ID': '9',
        'SLURM_JOB_ID': '9',
    }


def run_task_section(script: Path, task_id: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 # why: run the generated script under bash
        _task_section_command(script),
        check=False,
        env=_task_section_env(task_id),
        capture_output=True,
        text=True,
    )


def start_task_section(script: Path, task_id: str) -> subprocess.Popen[str]:
    return subprocess.Popen(  # noqa: S603 # why: run the generated script under bash
        _task_section_command(script),
        env=_task_section_env(task_id),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def test_job_script_header_and_task(tmp_path: Path) -> None:
    slurm = SlurmSettings(
        partition='CHANGE_ME',
        time='1:00:00',
        qos='normal',
        gpus=1,
        sbatch_extra=(('exclude', 'n1'),),
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


def test_job_script_writes_gpu_type(tmp_path: Path) -> None:
    script = _script(tmp_path, slurm=SlurmSettings(partition='p', gpus=2, gpu_type='a100'))
    assert '#SBATCH --gres=gpu:a100:2' in script.splitlines()


def test_job_script_ignores_gpu_type_without_gpus(tmp_path: Path) -> None:
    script = _script(tmp_path, slurm=SlurmSettings(partition='p', gpu_type='a100'))
    assert 'gres' not in script


def test_job_script_writes_setup_after_env_before_tasks(tmp_path: Path) -> None:
    slurm = SlurmSettings(partition='p', setup=('module load cuda', 'source /x/env.sh'))
    lines = _script(tmp_path, slurm=slurm, env={'A': 'b'}).splitlines()
    assert lines.index('module load cuda') == lines.index('export A=b') + 1
    assert lines.index('source /x/env.sh') < next(
        index for index, line in enumerate(lines) if line.startswith('for _ in')
    )


def test_job_script_rejects_a_line_break_in_setup(tmp_path: Path) -> None:
    slurm = SlurmSettings(partition='p', setup=('a\nb',))
    with pytest.raises(ValueError, match='setup'):
        _script(tmp_path, slurm=slurm)


def test_sbatch_extra_replaces_and_removes_tslab_lines(tmp_path: Path) -> None:
    slurm = SlurmSettings(
        partition='p',
        time='1:00:00',
        sbatch_extra=(('no_requeue', False), ('time', '2:00:00'), ('exclusive', True)),
    )
    lines = _script(tmp_path, slurm=slurm).splitlines()
    assert [line for line in lines if line.startswith('#SBATCH --time')] == [
        '#SBATCH --time=2:00:00'
    ]
    assert '#SBATCH --no-requeue' not in lines
    assert '#SBATCH --exclusive' in lines


def test_sbatch_extra_rejects_a_line_break_in_a_value(tmp_path: Path) -> None:
    slurm = SlurmSettings(partition='p', sbatch_extra=(('comment', 'a\nb'),))
    with pytest.raises(ValueError, match='sbatch_extra'):
        _script(tmp_path, slurm=slurm)


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
        {'slurm': SlurmSettings(partition='p', time='1:00:00', sbatch_extra=(('x', 'a\nrm'),))},
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


def test_task_section_waits_for_a_late_tasks_file(tmp_path: Path) -> None:
    tasks_file = tmp_path / 'jobs' / '1.tasks'
    script = write_script(tmp_path, tasks_file, write_tasks=False)
    process = start_task_section(script, task_id='3')
    time.sleep(2)
    tasks_file.parent.mkdir()
    tasks_file.write_text('\n'.join(TASKS_LINES) + '\n')
    out, _ = process.communicate(timeout=30)
    assert process.returncode == 0
    assert out.splitlines()[-1] == '3'


def test_sbatch_env_drops_job_variables_and_keeps_slurm_conf() -> None:
    environ = {
        'PATH': '/bin',
        'SLURM_JOB_ID': '1',
        'SLURM_CONF': '/etc/slurm.conf',
        'SBATCH_PARTITION': 'p',
        'SRUN_X': '1',
        'SLURMD_NODENAME': 'n',
        'SALLOC_X': '1',
        'RANK': '0',
        'MASTER_ADDR': 'h',
        'LOCAL_WORLD_SIZE': '2',
    }
    assert _sbatch_env(environ=environ) == {'PATH': '/bin', 'SLURM_CONF': '/etc/slurm.conf'}


def test_sbatch_passes_the_clean_env_and_leaves_os_environ(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv('SLURM_JOB_ID', '1')
    seen: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen.update(kwargs)
        return subprocess.CompletedProcess(command, 0, stdout='42;c\n', stderr='')

    monkeypatch.setattr(subprocess, 'run', fake_run)
    assert sbatch(script=tmp_path / 'x.sbatch') == '42'
    assert isinstance(seen['env'], dict)
    assert 'SLURM_JOB_ID' not in seen['env']
    assert os.environ['SLURM_JOB_ID'] == '1'


@pytest.mark.parametrize(
    ('key', 'expected'),
    [
        ('123_[4-7,9%10]', ['123_4', '123_5', '123_6', '123_7', '123_9']),
        ('123_[5]', ['123_5']),
        ('123_4', ['123_4']),
        ('123', ['123']),
    ],
)
def test_expand_array_key(key: str, expected: list[str]) -> None:
    assert _expand_array_key(key=key) == expected


def test_query_job_states_expands_pending_arrays_from_sacct(fake: FakeSlurm) -> None:
    fake.set_answers('sacct', [{'stdout': '123_[4-5%2]|PENDING\n'}])
    states = query_job_states(job_ids=['123_4', '123_5'])
    assert states == {'123_4': 'PENDING', '123_5': 'PENDING'}

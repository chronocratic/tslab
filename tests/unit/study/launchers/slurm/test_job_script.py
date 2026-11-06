"""SLURM study job scripts."""

import os
from pathlib import Path
import shlex
import subprocess
import time

import pytest

from tslab.study.code_snapshot import CodeSnapshot
from tslab.study.launchers.slurm.job_script import foreground_script, job_script
from tslab.study.machine import SlurmSettings

TASKS_LINES = [
    '3 1 ' + shlex.join(['python3', '-c', 'print(3)']),
    # why: the arguments hold a space and an apostrophe, the quoting round trip
    '17 2 ' + shlex.join(['python3', '-c', 'import sys; print(sys.argv[1:])', 'a b', "it's"]),
]


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


def _task_section_env(*, script: Path, task_id: str) -> dict[str, str]:
    bin_dir = script.parent / 'bin'
    bin_dir.mkdir(exist_ok=True)
    srun = bin_dir / 'srun'
    srun.write_text('#!/bin/sh\nwhile [ "${1#--}" != "$1" ]; do shift; done\nexec "$@"\n')
    srun.chmod(0o755)
    return {
        **os.environ,
        'PATH': f'{bin_dir}:{os.environ["PATH"]}',
        'SLURM_ARRAY_TASK_ID': task_id,
        'SLURM_ARRAY_JOB_ID': '9',
        'SLURM_JOB_ID': '9',
    }


def run_task_section(script: Path, task_id: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 # why: run the generated script under bash
        _task_section_command(script),
        check=False,
        env=_task_section_env(script=script, task_id=task_id),
        capture_output=True,
        text=True,
    )


def start_task_section(script: Path, task_id: str) -> subprocess.Popen[str]:
    return subprocess.Popen(  # noqa: S603 # why: run the generated script under bash
        _task_section_command(script),
        env=_task_section_env(script=script, task_id=task_id),
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


@pytest.mark.parametrize(
    ('name', 'slurm', 'indices'),
    [
        ('array', SlurmSettings(partition='compute', time='1:00:00'), (0, 1, 3)),
        ('followup', SlurmSettings(partition='compute', time='1:00:00'), None),
        (
            'gpus-4',
            SlurmSettings(partition='compute', time='1:00:00', gpus=4, gpu_type='a100'),
            (0,),
        ),
        ('signal-zero', SlurmSettings(partition='compute', time='1:00:00', signal_seconds=0), (0,)),
        (
            'sbatch-extra',
            SlurmSettings(
                partition='compute',
                time='1:00:00',
                sbatch_extra=(('no_requeue', False), ('time', '2:00:00'), ('exclusive', True)),
            ),
            (0,),
        ),
        (
            'node-local-tmp-false',
            SlurmSettings(partition='compute', time='1:00:00', node_local_tmp=False),
            (0,),
        ),
    ],
)
def test_job_script_is_unchanged(
    name: str, slurm: SlurmSettings, indices: tuple[int, ...] | None
) -> None:
    actual = job_script(
        job_name='train',
        slurm=slurm,
        indices=indices,
        tasks_file=Path('/project/jobs/0.tasks'),
        log_pattern=Path('/project/logs/train_%a.out'),
        project_root=Path('/project'),
        env={'EXAMPLE': 'a b'},
    )
    expected = Path(__file__).with_name('job_scripts') / f'{name}.sbatch'
    assert actual == expected.read_text()


def _golden_code_snapshot() -> CodeSnapshot:
    return CodeSnapshot(
        folder=Path('/snapshots/ab12cd34'),
        hash='ab12cd34ef56ab12',
        commit='c0ffee00',
        dirty=False,
        untracked_copied=False,
    )


def test_job_script_with_code_snapshot_matches_golden() -> None:
    actual = job_script(
        job_name='train',
        slurm=SlurmSettings(partition='compute', time='1:00:00'),
        indices=(0, 1, 3),
        tasks_file=Path('/project/jobs/0.tasks'),
        log_pattern=Path('/project/logs/train_%a.out'),
        project_root=Path('/project'),
        env={'EXAMPLE': 'a b'},
        code_snapshot=_golden_code_snapshot(),
    )
    expected = Path(__file__).with_name('job_scripts') / 'code-snapshot.sbatch'
    assert actual == expected.read_text()


def test_job_script_code_snapshot_comes_after_setup() -> None:
    # why: the snapshot lines come after the setup lines, so a setup that sets
    # PYTHONPATH is kept behind the snapshot
    slurm = SlurmSettings(partition='p', setup=('export PYTHONPATH=/setup/path',))
    lines = job_script(
        job_name='train',
        slurm=slurm,
        indices=(0,),
        tasks_file=Path('/project/jobs/0.tasks'),
        log_pattern=Path('/project/logs/train_%a.out'),
        project_root=Path('/project'),
        env={},
        code_snapshot=_golden_code_snapshot(),
    ).splitlines()
    assert lines.index('export PYTHONPATH=/setup/path') < lines.index(
        'export PYTHONPATH=/snapshots/ab12cd34/src${PYTHONPATH:+:$PYTHONPATH}'
    )


def test_job_script_without_code_snapshot_is_unchanged() -> None:
    actual = job_script(
        job_name='train',
        slurm=SlurmSettings(partition='compute', time='1:00:00'),
        indices=(0, 1, 3),
        tasks_file=Path('/project/jobs/0.tasks'),
        log_pattern=Path('/project/logs/train_%a.out'),
        project_root=Path('/project'),
        env={'EXAMPLE': 'a b'},
    )
    expected = Path(__file__).with_name('job_scripts') / 'array.sbatch'
    assert actual == expected.read_text()


def test_chained_script_checks_run_state_after_srun(tmp_path: Path) -> None:
    text = job_script(
        job_name='train',
        slurm=SlurmSettings(partition='compute', time='1:00:00'),
        indices=(0, 1),
        tasks_file=tmp_path / '0.tasks',
        log_pattern=tmp_path / 'logs' / 'train_%a.out',
        project_root=tmp_path,
        env={},
        chain_check=(
            'uv',
            'run',
            'tslab',
            'study',
            'chain-exit',
            'pkg.studies.chain:study',
            '--machine',
            'm.toml',
            '--job-file',
            '3',
        ),
    )

    assert '#SBATCH --kill-on-invalid-dep=yes' in text
    assert text.index('set +e') < text.index('srun --unbuffered')
    command = (
        'uv run tslab study chain-exit pkg.studies.chain:study --machine m.toml '
        '--job-file 3 --index ${SLURM_ARRAY_TASK_ID}'
    )
    assert text.index(command) > text.index('srun --unbuffered')
    assert text.rstrip().endswith('exit "$CHECK_EXIT"')


def _chained_text(*, requeue: bool | str) -> str:
    return job_script(
        job_name='train',
        slurm=SlurmSettings(partition='compute', time='1:00:00'),
        indices=(0, 1),
        tasks_file=Path('/p/0.tasks'),
        log_pattern=Path('/p/train_%a.out'),
        project_root=Path('/p'),
        env={},
        chain_check=('chain-check',),
        requeue=requeue,
    )


def test_requeue_mode_writes_requeue_not_no_requeue() -> None:
    text = _chained_text(requeue=True)
    assert '#SBATCH --requeue\n' in text
    assert '--no-requeue' not in text


def test_expedite_requeue_is_written_when_asked() -> None:
    assert '#SBATCH --requeue=expedite' in _chained_text(requeue='expedite')


def test_other_modes_keep_no_requeue() -> None:
    text = _chained_text(requeue=False)
    assert '#SBATCH --no-requeue' in text
    assert '--requeue' not in text


def test_requeue_header_adds_the_restart_count() -> None:
    assert '${SLURM_RESTART_COUNT:-0}' in _chained_text(requeue=True)
    assert 'SLURM_RESTART_COUNT' not in _chained_text(requeue=False)


def test_chain_check_runs_after_srun_failure_and_preserves_its_exit_code(tmp_path: Path) -> None:
    tasks_file = tmp_path / '0.tasks'
    tasks_file.write_text('3 1 true\n')
    script = tmp_path / 'job.sbatch'
    script.write_text(
        job_script(
            job_name='train',
            slurm=SlurmSettings(partition='compute', time='1:00:00'),
            indices=(3,),
            tasks_file=tasks_file,
            log_pattern=tmp_path / 'logs' / 'train_%a.out',
            project_root=tmp_path,
            env={},
            chain_check=('chain-check',),
        )
    )
    environment = _task_section_env(script=script, task_id='3')
    srun = script.parent / 'bin' / 'srun'
    srun.write_text('#!/bin/sh\nexit 9\n')
    checker_called = tmp_path / 'checker-called'
    checker = script.parent / 'bin' / 'chain-check'
    checker.write_text(f'#!/bin/sh\ntouch {shlex.quote(str(checker_called))}\nexit 0\n')
    checker.chmod(0o755)

    result = subprocess.run(  # noqa: S603 # why: the test runs the generated script under bash
        _task_section_command(script), check=False, env=environment, capture_output=True, text=True
    )

    assert result.returncode == 9
    assert checker_called.is_file()


def test_chain_check_exit_code_becomes_job_exit_code(tmp_path: Path) -> None:
    tasks_file = tmp_path / '0.tasks'
    tasks_file.write_text('3 1 true\n')
    script = tmp_path / 'job.sbatch'
    script.write_text(
        job_script(
            job_name='train',
            slurm=SlurmSettings(partition='compute', time='1:00:00'),
            indices=(3,),
            tasks_file=tasks_file,
            log_pattern=tmp_path / 'logs' / 'train_%a.out',
            project_root=tmp_path,
            env={},
            chain_check=('chain-check',),
        )
    )
    environment = _task_section_env(script=script, task_id='3')
    checker_called = tmp_path / 'checker-called'
    checker = script.parent / 'bin' / 'chain-check'
    checker.write_text(f'#!/bin/sh\ntouch {shlex.quote(str(checker_called))}\nexit 7\n')
    checker.chmod(0o755)

    result = subprocess.run(  # noqa: S603 # why: the test runs the generated script under bash
        _task_section_command(script), check=False, env=environment, capture_output=True, text=True
    )

    assert result.returncode == 7
    assert checker_called.is_file()


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


def test_job_script_requests_one_task_per_gpu_on_one_node(tmp_path: Path) -> None:
    lines = _script(tmp_path, slurm=SlurmSettings(partition='p', gpus=4)).splitlines()
    assert '#SBATCH --nodes=1' in lines
    assert '#SBATCH --ntasks-per-node=4' in lines


def test_job_script_without_gpus_runs_one_task(tmp_path: Path) -> None:
    lines = _script(tmp_path, slurm=SlurmSettings(partition='p')).splitlines()
    assert '#SBATCH --ntasks-per-node=1' in lines


def test_job_script_signals_usr2_before_the_time_limit(tmp_path: Path) -> None:
    lines = _script(tmp_path, slurm=SlurmSettings(partition='p')).splitlines()
    assert '#SBATCH --signal=USR2@120' in lines


def test_job_script_without_signal_when_signal_seconds_is_zero(tmp_path: Path) -> None:
    lines = _script(tmp_path, slurm=SlurmSettings(partition='p', signal_seconds=0)).splitlines()
    assert not any(line.startswith('#SBATCH --signal=') for line in lines)


def test_task_runs_through_srun(tmp_path: Path) -> None:
    text = _script(tmp_path, slurm=SlurmSettings(partition='p'))
    assert 'srun --unbuffered --kill-on-bad-exit=1 bash -c "exec $TASK_LINE"' in text
    assert 'eval "$TASK_LINE"' not in text


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


def test_foreground_script_has_env_and_setup_but_no_srun(tmp_path: Path) -> None:
    text = foreground_script(
        slurm=SlurmSettings(partition='p', setup=('module load x',)),
        project_root=tmp_path,
        env={'A': '1'},
        command=('python', '-m', 'm', '--x', 'a b'),
    )
    assert 'module load x' in text
    assert 'export A=1' in text
    assert text.rstrip().endswith("exec python -m m --x 'a b'")
    assert 'srun' not in text
    assert '#SBATCH' not in text
    assert 'TMPDIR' not in text


def test_foreground_script_orders_like_the_job_script(tmp_path: Path) -> None:
    text = foreground_script(
        slurm=SlurmSettings(partition='p', setup=('module load x',)),
        project_root=Path('/p q'),
        env={'A': 'b c'},
        command=('python', 'x.py'),
    )
    lines = text.splitlines()
    assert lines[0] == '#!/bin/bash'
    assert lines[1] == 'set -euo pipefail'
    assert lines[2] == "cd '/p q'"
    assert lines[3] == 'export PYTHONUNBUFFERED=1'
    assert lines[4] == "export A='b c'"
    assert lines[5] == 'module load x'
    assert lines[6] == 'exec python x.py'
    assert len(lines) == 7


def test_foreground_script_rejects_a_line_break_in_setup(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='setup'):
        foreground_script(
            slurm=SlurmSettings(partition='p', setup=('a\nb',)),
            project_root=tmp_path,
            env={},
            command=('python', 'x.py'),
        )


def test_foreground_script_rejects_bad_env_names(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='environment variable name'):
        foreground_script(
            slurm=SlurmSettings(partition='p'),
            project_root=tmp_path,
            env={'A B': 'v'},
            command=('python', 'x.py'),
        )

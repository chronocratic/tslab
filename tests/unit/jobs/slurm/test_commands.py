from datetime import timedelta
import os
from pathlib import Path
import subprocess

import pytest

from tests.unit.jobs.slurm.fake_slurm import FakeSlurm, install_fake_slurm
from tslab.jobs.slurm import (
    commands,
    query_expected_starts,
    query_job_states,
    query_job_usage,
    sbatch,
    scancel,
    SlurmError,
    SlurmJobUsage,
    SlurmQueueEntry,
    submission_environment,
)
from tslab.jobs.slurm.commands import _expand_array_key, _run, QOS_WAIT_CAP_SECONDS


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeSlurm:
    return install_fake_slurm(tmp_path=tmp_path, monkeypatch=monkeypatch, answers={})


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


def test_sbatch_gives_up_after_the_qos_cap(fake: FakeSlurm, tmp_path: Path) -> None:
    fake.set_answers('sbatch', [{'exit': 1, 'stderr': 'Job violates accounting/QOS policy'}])
    delays: list[float] = []
    with pytest.raises(SlurmError, match='sacctmgr show qos'):
        sbatch(script=tmp_path / 'x.sbatch', sleep=delays.append)
    assert sum(delays) <= QOS_WAIT_CAP_SECONDS
    assert sum(delays) + 900 > QOS_WAIT_CAP_SECONDS


def test_sbatch_logs_each_qos_wait(
    fake: FakeSlurm, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    fake.set_answers(
        'sbatch',
        [{'exit': 1, 'stderr': 'error: QOSMaxSubmitJobPerUserLimit'}] * 2
        + [{'exit': 0, 'stdout': '7;c'}],
    )
    sbatch(script=tmp_path / 'x.sbatch', sleep=lambda _: None)
    waits = [r for r in caplog.records if 'QOSMaxSubmitJobPerUserLimit' in r.getMessage()]
    assert len(waits) == 2
    assert 'retrying in 10' in waits[0].getMessage()


def test_sbatch_other_errors_raise(fake: FakeSlurm, tmp_path: Path) -> None:
    fake.set_answers('sbatch', [{'exit': 1, 'stderr': 'error: invalid partition'}])
    with pytest.raises(SlurmError, match='invalid partition'):
        sbatch(script=tmp_path / 'x.sbatch', sleep=lambda _: None)


def test_sbatch_timeout_raises_without_retry(
    fake: FakeSlurm, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # why: the fake is a Python script; its startup must fit inside the timeout
    monkeypatch.setattr(commands, 'SUBMIT_TIMEOUT_SECONDS', 2.0)
    fake.set_answers('sbatch', [{'exit': 0, 'stdout': '1;c', 'sleep': 5}])
    with pytest.raises(SlurmError, match='did not answer'):
        sbatch(script=tmp_path / 'x.sbatch', sleep=lambda _: None)
    assert len(fake.calls('sbatch')) == 1


def test_run_raises_on_timeout() -> None:
    with pytest.raises(SlurmError, match=r'did not answer within 0.2 seconds'):
        _run(command=['sleep', '5'], timeout=0.2)


def test_scancel_with_signal_passes_signal_flag(fake: FakeSlurm) -> None:
    scancel(job_ids=['1000_0'], signal='USR2')
    assert fake.calls('scancel') == [['--signal=USR2', '1000_0']]


def test_scancel_without_signal_is_unchanged(fake: FakeSlurm) -> None:
    scancel(job_ids=['1000_0'])
    assert fake.calls('scancel') == [['1000_0']]


def test_scancel_empty_ids_calls_nothing(fake: FakeSlurm) -> None:
    scancel(job_ids=[], signal='USR2')
    assert fake.calls('scancel') == []


def test_query_expected_starts_maps_na_to_none(fake: FakeSlurm) -> None:
    fake.set_answers(
        'squeue', [{'exit': 0, 'stdout': '1000_0|2026-11-01T10:00:00|\n1000_1|N/A|\n'}]
    )
    assert query_expected_starts(job_ids=['1000_0', '1000_1']) == {
        '1000_0': SlurmQueueEntry(start='2026-11-01T10:00:00', reason=None),
        '1000_1': SlurmQueueEntry(start=None, reason=None),
    }
    assert fake.calls('squeue') == [
        ['--array', '--start', '--jobs', '1000_0,1000_1', '--noheader', '--format=%i|%S|%r']
    ]


def test_query_expected_starts_empty_value_is_none(fake: FakeSlurm) -> None:
    fake.set_answers('squeue', [{'exit': 0, 'stdout': '1000_0|\n'}])
    assert query_expected_starts(job_ids=['1000_0']) == {
        '1000_0': SlurmQueueEntry(start=None, reason=None)
    }


def test_query_expected_starts_expands_array_ranges(fake: FakeSlurm) -> None:
    fake.set_answers('squeue', [{'exit': 0, 'stdout': '1000_[2-3]|2026-11-01T10:00:00|\n'}])
    assert query_expected_starts(job_ids=['1000_2', '1000_3']) == {
        '1000_2': SlurmQueueEntry(start='2026-11-01T10:00:00', reason=None),
        '1000_3': SlurmQueueEntry(start='2026-11-01T10:00:00', reason=None),
    }


def test_query_expected_starts_reads_the_reason(fake: FakeSlurm) -> None:
    fake.set_answers(
        'squeue', [{'exit': 0, 'stdout': '1000_[0-1]|N/A|ReqNodeNotAvail, UnavailableNodes:n1\n'}]
    )
    entries = query_expected_starts(job_ids=['1000_0', '1000_1'])
    assert entries['1000_1'] == SlurmQueueEntry(
        start=None, reason='ReqNodeNotAvail, UnavailableNodes:n1'
    )
    assert fake.calls('squeue')[0][-1] == '--format=%i|%S|%r'


def test_query_expected_starts_empty_ids_calls_nothing(fake: FakeSlurm) -> None:
    assert query_expected_starts(job_ids=[]) == {}
    assert fake.calls('squeue') == []


def test_query_expected_starts_raises_on_failure(
    fake: FakeSlurm, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(commands, '_sleep', lambda _: None)
    fake.set_answers('squeue', [{'exit': 1, 'stderr': 'slurm_load_jobs error: Socket timed out'}])
    with pytest.raises(SlurmError, match='Socket timed out'):
        query_expected_starts(job_ids=['1000_0'])


def test_query_expected_starts_accepts_squeue_not_knowing_a_job(fake: FakeSlurm) -> None:
    fake.set_answers(
        'squeue', [{'exit': 1, 'stderr': 'slurm_load_jobs error: Invalid job id specified'}]
    )
    assert query_expected_starts(job_ids=['1000_0']) == {}


def test_query_job_usage_folds_steps_into_task(fake: FakeSlurm) -> None:
    fake.set_answers(
        'sacct',
        [
            {
                'exit': 0,
                'stdout': (
                    '1000_2|COMPLETED|01:00:00|02:00:00|00:50:00.5||4|billing=4,cpu=4,gres/gpu=1\n'
                    '1000_2.batch|COMPLETED|01:00:00||00:01:00|100M|4|\n'
                    '1000_2.0|COMPLETED|00:59:00||00:49:00|3G|4|\n'
                ),
            }
        ],
    )
    usage = query_job_usage(job_ids=['1000_2'])['1000_2']
    assert isinstance(usage, SlurmJobUsage)
    assert usage.state == 'COMPLETED'
    assert usage.max_rss == 3 * 1024**3
    assert usage.gpus == 1
    assert usage.alloc_cpus == 4
    assert usage.elapsed == timedelta(hours=1)
    assert usage.time_limit == timedelta(hours=2)
    assert usage.total_cpu == timedelta(minutes=50)


def test_query_job_usage_running_job_has_no_rss(fake: FakeSlurm) -> None:
    fake.set_answers('sacct', [{'exit': 0, 'stdout': '1000_2|RUNNING|00:10:00|02:00:00|||4|\n'}])
    usage = query_job_usage(job_ids=['1000_2'])['1000_2']
    assert usage.state == 'RUNNING'
    assert usage.max_rss is None
    assert usage.total_cpu is None
    assert usage.elapsed == timedelta(minutes=10)


def test_query_job_usage_keeps_first_word_of_state(fake: FakeSlurm) -> None:
    fake.set_answers(
        'sacct', [{'exit': 0, 'stdout': '1000_2|CANCELLED by 123|01:00:00|02:00:00|||4|\n'}]
    )
    assert query_job_usage(job_ids=['1000_2'])['1000_2'].state == 'CANCELLED'


def test_query_job_usage_empty_ids_calls_nothing(fake: FakeSlurm) -> None:
    assert query_job_usage(job_ids=[]) == {}
    assert fake.calls('sacct') == []


def test_query_job_usage_raises_on_failure(
    fake: FakeSlurm, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(commands, '_sleep', lambda _: None)
    fake.set_answers('sacct', [{'exit': 1, 'stderr': 'sacct: error: Problem talking to slurmdbd'}])
    with pytest.raises(SlurmError, match='slurmdbd'):
        query_job_usage(job_ids=['1000_2'])


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


def test_query_job_states_raises_when_squeue_fails(
    fake: FakeSlurm, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(commands, '_sleep', lambda _: None)
    fake.set_answers('squeue', [{'exit': 1, 'stderr': 'slurm_load_jobs error: Socket timed out'}])
    with pytest.raises(SlurmError, match='Socket timed out'):
        query_job_states(job_ids=['7'])


def test_query_job_states_raises_when_sacct_fails(
    fake: FakeSlurm, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(commands, '_sleep', lambda _: None)
    fake.set_answers('squeue', [{'exit': 0, 'stdout': '7_0|RUNNING\n'}])
    fake.set_answers('sacct', [{'exit': 1, 'stderr': 'sacct: error: Problem talking to slurmdbd'}])
    with pytest.raises(SlurmError, match='slurmdbd'):
        query_job_states(job_ids=['7'])


def test_query_job_states_retries_a_failed_query(
    fake: FakeSlurm, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(commands, '_sleep', lambda _: None)
    fake.set_answers(
        'squeue',
        [{'exit': 1, 'stderr': 'Socket timed out'}, {'exit': 0, 'stdout': '1000_0|RUNNING\n'}],
    )
    assert query_job_states(job_ids=['1000_0']) == {'1000_0': 'RUNNING'}
    assert len(fake.calls('squeue')) == 2


def test_query_job_states_raises_after_three_failures(
    fake: FakeSlurm, monkeypatch: pytest.MonkeyPatch
) -> None:
    delays: list[float] = []
    monkeypatch.setattr(commands, '_sleep', delays.append)
    fake.set_answers('squeue', [{'exit': 1, 'stderr': 'Socket timed out'}])
    with pytest.raises(SlurmError, match='Socket timed out'):
        query_job_states(job_ids=['1000_0'])
    assert len(fake.calls('squeue')) == 3
    assert delays == [5.0, 15.0]


def test_query_job_states_accepts_squeue_not_knowing_a_job(fake: FakeSlurm) -> None:
    fake.set_answers(
        'squeue', [{'exit': 1, 'stderr': 'slurm_load_jobs error: Invalid job id specified'}]
    )
    fake.set_answers('sacct', [{'exit': 0, 'stdout': '6_0|COMPLETED\n'}])
    assert query_job_states(job_ids=['6']) == {'6_0': 'COMPLETED'}


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


def test_submission_environment_drops_job_variables_and_keeps_slurm_conf() -> None:
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
    assert submission_environment(environ=environ) == {
        'PATH': '/bin',
        'SLURM_CONF': '/etc/slurm.conf',
    }


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


def test_sbatch_cleans_the_given_environ(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('TSLAB_MARKER', '1')
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    sbatch_script = bin_dir / 'sbatch'
    sbatch_script.write_text('#!/bin/sh\nenv | sort > "$ENV_OUT"\necho "42;cluster"\n')
    sbatch_script.chmod(0o755)
    monkeypatch.setenv('PATH', f'{bin_dir}:{os.environ["PATH"]}')
    job_id = sbatch(
        script=tmp_path / 'x.sbatch',
        environ={
            'PATH': os.environ['PATH'],
            'SLURM_JOB_ID': '1',
            'KEEP': 'x',
            'ENV_OUT': str(tmp_path / 'env.txt'),
        },
        sleep=lambda _seconds: None,
    )
    assert job_id == '42'
    lines = (tmp_path / 'env.txt').read_text().splitlines()
    assert 'KEEP=x' in lines
    assert not any(line.startswith('SLURM_JOB_ID=') for line in lines)
    assert 'TSLAB_MARKER=1' not in lines

"""Tests for the failure hints of the SLURM scheduler states."""

from tslab.study.launchers.common.types import JobUsage
from tslab.study.launchers.slurm.hints import state_hint
from tslab.study.machine import SlurmSettings


def _oom_usage(*, max_rss: int | None) -> JobUsage:
    return JobUsage(
        state='OUT_OF_MEMORY',
        elapsed=None,
        time_limit=None,
        total_cpu=None,
        max_rss=max_rss,
        alloc_cpus=None,
        gpus=0,
    )


_RESOURCES = SlurmSettings(partition='p', mem_per_cpu='2G')


def test_hint_for_oom_with_numbers() -> None:
    hint = state_hint(
        job_state='OUT_OF_MEMORY', usage=_oom_usage(max_rss=3 * 1024**3), resources=_RESOURCES
    )
    assert hint == 'raise mem_per_cpu; peak 3072M of 2048M requested'


def test_hint_for_oom_without_numbers() -> None:
    hint = state_hint(job_state='OUT_OF_MEMORY', usage=None, resources=_RESOURCES)
    assert hint == 'raise mem_per_cpu'


def test_hint_for_timeout() -> None:
    assert state_hint(job_state='TIMEOUT', usage=None, resources=_RESOURCES) == (
        'raise time, or check that the timeout signal is on (signal_seconds), '
        'or lower job_time_share so the stages after training fit'
    )


def test_hint_for_preempted_and_node_fail() -> None:
    for job_state in ('PREEMPTED', 'NODE_FAIL'):
        assert state_hint(job_state=job_state, usage=None, resources=_RESOURCES) == (
            'not caused by the code; resubmit'
        )


def test_no_hint_for_other_states() -> None:
    assert state_hint(job_state='FAILED', usage=None, resources=_RESOURCES) is None
    assert state_hint(job_state=None, usage=None, resources=_RESOURCES) is None

import pytest

from tslab.lab.jobs.slurm import job_ended, pending_reason_stuck


def test_a_job_has_ended_unless_it_is_unknown_queued_or_running() -> None:
    assert job_ended(job_state='FAILED')
    assert not job_ended(job_state=None)
    assert not job_ended(job_state='RUNNING')


@pytest.mark.parametrize(
    ('reason', 'stuck'),
    [
        ('PartitionTimeLimit', True),
        ('(PartitionTimeLimit)', True),
        ('Priority', False),
        ('ReqNodeNotAvail, UnavailableNodes:n1', False),
        (None, False),
    ],
)
def test_pending_reason_stuck(*, reason: str | None, stuck: bool) -> None:
    assert pending_reason_stuck(reason=reason) is stuck

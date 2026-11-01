from tslab.jobs.slurm import job_ended


def test_a_job_has_ended_unless_it_is_unknown_queued_or_running() -> None:
    assert job_ended(job_state='FAILED')
    assert not job_ended(job_state=None)
    assert not job_ended(job_state='RUNNING')

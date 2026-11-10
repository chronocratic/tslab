import pytest

from tslab.lightning.slurm import job_training_deadline

NOW = 1_000_000.0


def test_job_training_deadline_is_share_of_remaining_time() -> None:
    assert (
        job_training_deadline(
            share=0.5, environment={'SLURM_JOB_END_TIME': str(int(NOW) + 100)}, now=NOW
        )
        == NOW + 50
    )


def test_job_training_deadline_is_none_outside_slurm() -> None:
    assert job_training_deadline(share=0.5, environment={}, now=NOW) is None


def test_job_training_deadline_is_now_when_job_has_ended() -> None:
    environment = {'SLURM_JOB_END_TIME': str(int(NOW) - 10)}
    assert job_training_deadline(share=0.5, environment=environment, now=NOW) == NOW


@pytest.mark.parametrize('share', [0.0, -0.5, 1.5])
def test_job_time_share_must_be_in_range(*, share: float) -> None:
    with pytest.raises(ValueError, match='job_time_share must be in'):
        job_training_deadline(share=share, environment={}, now=NOW)

from datetime import timedelta
import logging

import pytest

from tslab.lightning.slurm import job_training_deadline, parse_slurm_duration

NOW = 1_000_000.0


@pytest.mark.parametrize(
    ('text', 'expected'),
    [
        ('00:00:00', timedelta(0)),
        ('90', timedelta(minutes=90)),
        ('30:00', timedelta(minutes=30)),
        ('0:30', timedelta(seconds=30)),
        ('1:30:15', timedelta(hours=1, minutes=30, seconds=15)),
        ('1-06:00:00', timedelta(days=1, hours=6)),
        ('2-12:00', timedelta(days=2, hours=12)),
        ('2-12', timedelta(days=2, hours=12)),
        ('1-00:30:10', timedelta(days=1, minutes=30, seconds=10)),
    ],
)
def test_parse_slurm_duration(*, text: str, expected: timedelta) -> None:
    assert parse_slurm_duration(text=text) == expected


@pytest.mark.parametrize(
    'text', ['', '1:2:3:4', 'one:00', '-1:00', '1-', '1-2:3:4:5', '2:-5', '1:-30:00']
)
def test_parse_slurm_duration_rejects_other_formats(*, text: str) -> None:
    with pytest.raises(ValueError, match='sbatch'):
        parse_slurm_duration(text=text)


def test_job_training_deadline_is_share_of_remaining_time() -> None:
    assert (
        job_training_deadline(
            share=0.5, environment={'SLURM_JOB_END_TIME': str(int(NOW) + 100)}, now=NOW
        )
        == NOW + 50
    )


def test_job_training_deadline_is_none_outside_slurm() -> None:
    assert job_training_deadline(share=0.5, environment={}, now=NOW) is None


def test_invalid_end_time_is_ignored(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        deadline = job_training_deadline(
            share=0.5, environment={'SLURM_JOB_END_TIME': 'soon'}, now=NOW
        )
    assert deadline is None
    assert 'SLURM_JOB_END_TIME' in caplog.text


def test_job_training_deadline_is_now_when_job_has_ended() -> None:
    environment = {'SLURM_JOB_END_TIME': str(int(NOW) - 10)}
    assert job_training_deadline(share=0.5, environment=environment, now=NOW) == NOW


@pytest.mark.parametrize('share', [0.0, -0.5, 1.5])
def test_job_time_share_must_be_in_range(*, share: float) -> None:
    with pytest.raises(ValueError, match='job_time_share must be in'):
        job_training_deadline(share=share, environment={}, now=NOW)

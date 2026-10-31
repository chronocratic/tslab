from datetime import timedelta
import logging

import pytest

from tslab.jobs.slurm import parse_slurm_duration, remaining_job_seconds

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


def test_remaining_job_seconds_is_none_without_end_time() -> None:
    assert remaining_job_seconds(environment={}, now=NOW) is None


def test_remaining_job_seconds_returns_time_until_end() -> None:
    assert remaining_job_seconds(environment={'SLURM_JOB_END_TIME': '1000100'}, now=NOW) == 100.0


def test_remaining_job_seconds_never_negative() -> None:
    assert remaining_job_seconds(environment={'SLURM_JOB_END_TIME': '999990'}, now=NOW) == 0.0


def test_remaining_job_seconds_ignores_invalid_end_time(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        remaining = remaining_job_seconds(environment={'SLURM_JOB_END_TIME': 'soon'}, now=NOW)
    assert remaining is None
    assert 'SLURM_JOB_END_TIME' in caplog.text

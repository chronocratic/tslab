from datetime import timedelta
import logging

import pytest

from tslab.lightning.slurm import parse_slurm_duration, resolve_training_time

NOW = 1_000_000.0
ONE_HOUR_LEFT = {'SLURM_JOB_END_TIME': str(int(NOW) + 3600)}


@pytest.mark.parametrize(
    ('text', 'expected'),
    [
        ('00:00:00', timedelta(0)),
        ('24:00', timedelta(hours=24)),
        ('1:30:15', timedelta(hours=1, minutes=30, seconds=15)),
        ('1-06:00:00', timedelta(days=1, hours=6)),
        ('2-12:00', timedelta(days=2, hours=12)),
    ],
)
def test_parse_slurm_duration(*, text: str, expected: timedelta) -> None:
    assert parse_slurm_duration(text=text) == expected


@pytest.mark.parametrize('text', ['', '90', '1:2:3:4', 'one:00', '-1:00'])
def test_parse_slurm_duration_rejects_other_formats(*, text: str) -> None:
    with pytest.raises(ValueError, match='HH:MM:SS'):
        parse_slurm_duration(text=text)


def test_no_limit_outside_slurm() -> None:
    assert resolve_training_time(explicit=None, fraction=0.5, environment={}, now=NOW) is None


def test_explicit_limit_outside_slurm() -> None:
    limit = resolve_training_time(explicit='02:00:00', fraction=0.5, environment={}, now=NOW)
    assert limit == timedelta(hours=2)


def test_fraction_of_the_remaining_job_time() -> None:
    limit = resolve_training_time(explicit=None, fraction=0.5, environment=ONE_HOUR_LEFT, now=NOW)
    assert limit == timedelta(minutes=30)


def test_explicit_limit_within_the_job_is_kept() -> None:
    limit = resolve_training_time(
        explicit='00:20:00', fraction=0.5, environment=ONE_HOUR_LEFT, now=NOW
    )
    assert limit == timedelta(minutes=20)


def test_explicit_limit_beyond_the_job_falls_back(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        limit = resolve_training_time(
            explicit='02:00:00', fraction=0.5, environment=ONE_HOUR_LEFT, now=NOW
        )
    assert limit == timedelta(minutes=30)
    assert 'exceeds the remaining SLURM job time' in caplog.text


def test_invalid_end_time_is_ignored(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        limit = resolve_training_time(
            explicit=None, fraction=0.5, environment={'SLURM_JOB_END_TIME': 'soon'}, now=NOW
        )
    assert limit is None
    assert 'SLURM_JOB_END_TIME' in caplog.text


def test_job_already_over_gives_zero() -> None:
    environment = {'SLURM_JOB_END_TIME': str(int(NOW) - 10)}
    limit = resolve_training_time(explicit=None, fraction=0.5, environment=environment, now=NOW)
    assert limit == timedelta(0)


@pytest.mark.parametrize('fraction', [0.0, -0.5, 1.5])
def test_fraction_must_be_in_range(*, fraction: float) -> None:
    with pytest.raises(ValueError, match='slurm_time_fraction'):
        resolve_training_time(explicit=None, fraction=fraction, environment={}, now=NOW)

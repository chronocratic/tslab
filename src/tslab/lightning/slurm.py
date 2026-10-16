"""Training time limit from the trainer settings and the SLURM job end time.

SLURM
exports the job's end as a Unix timestamp in ``SLURM_JOB_END_TIME``; training
gets a fraction of the time left, so the rest of the job can still run the
stages after training and write the checkpoint.
"""

from collections.abc import Mapping
from datetime import timedelta
import logging

__all__ = ['parse_slurm_duration', 'resolve_training_time']

_LOGGER = logging.getLogger(__name__)
_END_TIME_VARIABLE = 'SLURM_JOB_END_TIME'
_FORMAT_HINT = 'use "HH:MM", "HH:MM:SS" or "D-HH:MM:SS"'


def parse_slurm_duration(*, text: str) -> timedelta:
    """Parse a duration in sbatch ``--time`` format.

    Args:
        text: ``"HH:MM"``, ``"HH:MM:SS"``, ``"D-HH:MM"`` or ``"D-HH:MM:SS"``.

    Returns:
        The duration.

    Raises:
        ValueError: If ``text`` has another format.
    """
    days_text, separator, clock = text.partition('-')
    if not separator:
        days_text, clock = '0', text
    try:
        days = int(days_text)
        parts = [int(part) for part in clock.split(':')]
    except ValueError as error:
        message = f'invalid duration {text!r}; {_FORMAT_HINT}'
        raise ValueError(message) from error
    match parts:
        case [hours, minutes]:
            return timedelta(days=days, hours=hours, minutes=minutes)
        case [hours, minutes, seconds]:
            return timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
        case _:
            message = f'invalid duration {text!r}; {_FORMAT_HINT}'
            raise ValueError(message)


def _remaining_job_seconds(*, environment: Mapping[str, str], now: float) -> float | None:
    raw = environment.get(_END_TIME_VARIABLE)
    if raw is None:
        return None
    try:
        end_time = int(raw)
    except ValueError:
        _LOGGER.warning('ignoring %s=%r: not a Unix timestamp', _END_TIME_VARIABLE, raw)
        return None
    return max(0.0, end_time - now)


def resolve_training_time(
    *, explicit: str | None, fraction: float, environment: Mapping[str, str], now: float
) -> timedelta | None:
    """Return the training time limit, or ``None`` for no limit.

    Args:
        explicit: ``max_training_time`` from the settings, in sbatch format.
        fraction: Share of the remaining SLURM job time used for training.
        environment: Environment variables, usually ``os.environ``.
        now: Current Unix time, usually ``time.time()``.

    Returns:
        The explicit limit if it fits in the job; otherwise ``fraction`` of the
        remaining job time under SLURM; otherwise ``None``.

    Raises:
        ValueError: If ``fraction`` is not in ``(0, 1]`` or ``explicit`` is malformed.
    """
    if not 0 < fraction <= 1:
        message = f'slurm_time_fraction must be in (0, 1], got {fraction}'
        raise ValueError(message)
    remaining = _remaining_job_seconds(environment=environment, now=now)
    if explicit is not None:
        requested = parse_slurm_duration(text=explicit)
        if remaining is None or requested.total_seconds() <= remaining:
            return requested
        fallback = timedelta(seconds=int(fraction * remaining))
        _LOGGER.warning(
            'max_training_time %s exceeds the remaining SLURM job time (%.0f s); '
            'training for %.0f%% of it instead (%s)',
            explicit,
            remaining,
            fraction * 100,
            fallback,
        )
        return fallback
    if remaining is None:
        return None
    return timedelta(seconds=int(fraction * remaining))

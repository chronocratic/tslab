"""SLURM duration parsing and job-end time helpers."""

from collections.abc import Mapping
from datetime import timedelta
import logging

__all__ = [
    'format_slurm_duration',
    'parse_sacct_duration',
    'parse_slurm_duration',
    'remaining_job_seconds',
]

_LOGGER = logging.getLogger(__name__)
_END_TIME_VARIABLE = 'SLURM_JOB_END_TIME'
_FORMAT_HINT = 'use an sbatch --time format: "M", "M:S", "H:M:S", "D-H", "D-H:M" or "D-H:M:S"'

# why: what sacct prints in the duration fields when there is no duration to report
_SACCT_NON_VALUES = ('UNLIMITED', 'Partition_Limit', 'INVALID')


def parse_slurm_duration(*, text: str) -> timedelta:
    """Parse a duration in sbatch ``--time`` format.

    Args:
        text: ``"M"`` (minutes), ``"M:S"``, ``"H:M:S"``, ``"D-H"``, ``"D-H:M"`` or
            ``"D-H:M:S"``, with D days, H hours, M minutes and S seconds.

    Returns:
        The duration.

    Raises:
        ValueError: If ``text`` has another format or a negative part.
    """
    days_text, separator, clock = text.partition('-')
    has_days = bool(separator)
    if not has_days:
        clock = text
    try:
        days = int(days_text) if has_days else 0
        parts = [int(part) for part in clock.split(':')]
    except ValueError as error:
        message = f'invalid duration {text!r}; {_FORMAT_HINT}'
        raise ValueError(message) from error
    # With days the first clock part is hours; without, it is minutes (or hours for H:M:S).
    match has_days, parts:
        case True, [hours]:
            clock_delta = timedelta(hours=hours)
        case True, [hours, minutes]:
            clock_delta = timedelta(hours=hours, minutes=minutes)
        case True, [hours, minutes, seconds]:
            clock_delta = timedelta(hours=hours, minutes=minutes, seconds=seconds)
        case False, [minutes]:
            clock_delta = timedelta(minutes=minutes)
        case False, [minutes, seconds]:
            clock_delta = timedelta(minutes=minutes, seconds=seconds)
        case False, [hours, minutes, seconds]:
            clock_delta = timedelta(hours=hours, minutes=minutes, seconds=seconds)
        case _:
            message = f'invalid duration {text!r}; {_FORMAT_HINT}'
            raise ValueError(message)
    if days < 0 or any(part < 0 for part in parts):
        message = f'invalid duration {text!r}; {_FORMAT_HINT}, without negative parts'
        raise ValueError(message)
    return timedelta(days=days) + clock_delta


def parse_sacct_duration(*, text: str) -> timedelta | None:
    """Parse a duration from ``sacct``, which also prints words and fractions.

    Args:
        text: A duration such as ``"1-02:03:04"`` or ``"01:02.345"``, or
            ``""``, ``"UNLIMITED"``, ``"Partition_Limit"`` or ``"INVALID"``.

    Returns:
        The duration, with a fraction of a second dropped; ``None`` for an
        empty string and the non-duration values.
    """
    stripped = text.strip()
    if stripped in ('', *_SACCT_NON_VALUES):
        return None
    # why: sacct prints TotalCPU with a fraction of a second, sbatch --time does not
    whole, _, fraction = stripped.partition('.')
    if fraction:
        stripped = whole
    return parse_slurm_duration(text=stripped)


def format_slurm_duration(*, duration: timedelta) -> str:
    """Format a duration in sbatch ``--time`` format, rounded up to the minute.

    Args:
        duration: The duration to format.

    Returns:
        ``H:MM:SS`` below one day, ``D-HH:MM:SS`` from one day.
    """
    whole_seconds = duration // timedelta(seconds=1)
    if duration > whole_seconds * timedelta(seconds=1):
        whole_seconds += 1
    minutes = (int(whole_seconds) + 59) // 60
    days, remainder = divmod(minutes, 24 * 60)
    hours, mins = divmod(remainder, 60)
    if days:
        return f'{days}-{hours:02d}:{mins:02d}:00'
    return f'{hours}:{mins:02d}:00'


def remaining_job_seconds(*, environment: Mapping[str, str], now: float) -> float | None:
    """Return the seconds remaining before the SLURM job ends.

    Args:
        environment: Environment variables, usually ``os.environ``.
        now: Current Unix time, usually ``time.time()``.

    Returns:
        The seconds until the job end, or ``None`` when no valid end time is set.
    """
    raw = environment.get(_END_TIME_VARIABLE)
    if raw is None:
        return None
    try:
        end_time = int(raw)
    except ValueError:
        _LOGGER.warning('ignoring %s=%r: not a Unix timestamp', _END_TIME_VARIABLE, raw)
        return None
    return max(0.0, end_time - now)

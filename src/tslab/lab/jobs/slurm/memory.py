"""SLURM memory value parsing and formatting."""

import math
import re

__all__ = ['format_slurm_memory', 'parse_slurm_memory']

# why: the multiplier of each suffix in bytes, base 1024, as SLURM names them
_UNITS = {'K': 1024, 'M': 1024**2, 'G': 1024**3, 'T': 1024**4}
_MEGABYTES = 1024**2
_PATTERN = re.compile(r'^(?P<value>\d+(?:\.\d+)?)(?P<unit>[KMGT]?)$')


def parse_slurm_memory(*, text: str) -> int | None:
    """Parse a SLURM memory value into bytes.

    Args:
        text: A value such as ``"1.5G"`` or ``"4000"``; the suffixes are
            ``K M G T`` in base 1024, and a bare number is megabytes, as in
            sbatch ``--mem-per-cpu``.

    Returns:
        The size in bytes, or ``None`` for an empty string.

    Raises:
        ValueError: If ``text`` is not a memory value.
    """
    stripped = text.strip()
    if not stripped:
        return None
    match = _PATTERN.fullmatch(stripped)
    if match is None:
        message = f'invalid memory value {text!r}'
        raise ValueError(message)
    value = float(match['value'])
    unit = _UNITS[match['unit'] or 'M']
    return int(value * unit)


def format_slurm_memory(*, size: int) -> str:
    """Format a size in bytes as a SLURM memory value in whole megabytes.

    Args:
        size: The size in bytes.

    Returns:
        The size rounded up to whole megabytes, at least ``"1M"``.
    """
    megabytes = math.ceil(size / _MEGABYTES)
    return f'{max(megabytes, 1)}M'

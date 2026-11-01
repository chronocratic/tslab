"""The trailing lines of a log file."""

from collections import deque
from pathlib import Path

__all__ = ['tail_lines']


def tail_lines(*, path: Path, lines: int) -> list[str] | None:
    """Read the last lines of a file, or return ``None`` when it is missing.

    Args:
        path: The file to read.
        lines: Maximum number of trailing lines.

    Returns:
        The trailing lines without line endings, or ``None`` when missing.
    """
    try:
        with path.open(encoding='utf-8', errors='replace') as file:
            return [line.rstrip('\r\n') for line in deque(file, maxlen=lines)]
    except FileNotFoundError:
        return None

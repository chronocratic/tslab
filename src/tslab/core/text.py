"""Text helpers: error lines, class names, first lines and aligned tables."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import difflib
import logging

__all__ = [
    'TextTable',
    'describe_error',
    'first_line',
    'format_bytes',
    'format_duration',
    'format_index_ranges',
    'format_table',
    'log_table',
    'qualified_name',
    'unknown_key_message',
]


def describe_error(*, error: BaseException) -> str:
    """Return ``"ExceptionType: message"``, the form stored in ``status.json``.

    Args:
        error: Any exception.
    """
    return f'{type(error).__name__}: {error}'


def qualified_name(*, cls: type) -> str:
    """Return ``module.QualName`` of a class, the spelling used in files on disk.

    Args:
        cls: Any class.
    """
    return f'{cls.__module__}.{cls.__qualname__}'


def first_line(*, text: str) -> str:
    """Return the first line of ``text``, or ``''`` if it has none.

    Args:
        text: Any text, for example an exception message.
    """
    lines = text.splitlines()
    return lines[0] if lines else ''


def unknown_key_message(*, location: str, key: str, valid: Iterable[str]) -> str:
    """Return a message for an unknown key: the close valid key, then all valid keys.

    Args:
        location: Where the key was found, for example ``slurm``; empty for the top level.
        key: The key that is not valid.
        valid: The keys that are valid at that place.

    Returns:
        A one-line message that names the key and lists the valid keys.
    """
    names = sorted(valid)
    where = f'{location}.{key}' if location else key
    close = difflib.get_close_matches(key, names, n=1)
    hint = f"; did you mean '{close[0]}'?" if close else '.'
    return f'{where}: unknown key{hint} Valid keys: {", ".join(names)}'


def format_duration(*, seconds: float) -> str:
    """Format a length of time in seconds for display, as ``H:MM:SS``.

    Args:
        seconds: The length of time in seconds; the fraction is dropped.
    """
    total = int(seconds)
    return f'{total // 3600}:{(total % 3600) // 60:02d}:{total % 60:02d}'


def format_bytes(*, size: int) -> str:
    """Format a size in bytes for display, in the largest unit that keeps it at one or more.

    Args:
        size: The size in bytes.

    Returns:
        The size with one decimal, e.g. ``1.5 GiB``.
    """
    for unit, divisor in (('GiB', 1024**3), ('MiB', 1024**2), ('KiB', 1024)):
        if size >= divisor:
            return f'{size / divisor:.1f} {unit}'
    return f'{size / 1024:.1f} KiB'


def format_index_ranges(*, indices: Sequence[int]) -> str:
    """Format indices as sorted, deduplicated ranges, e.g. ``0-3,7,9-10``.

    Args:
        indices: The indices to format.

    Returns:
        The comma-separated ranges.

    Raises:
        ValueError: If ``indices`` is empty.
    """
    ordered = sorted(set(indices))
    if not ordered:
        message = 'indices must not be empty'
        raise ValueError(message)
    parts: list[str] = []
    start = previous = ordered[0]
    for index in ordered[1:]:
        if index == previous + 1:
            previous = index
            continue
        parts.append(f'{start}-{previous}' if previous > start else str(start))
        start = previous = index
    parts.append(f'{start}-{previous}' if previous > start else str(start))
    return ','.join(parts)


def format_table(
    *, header: Sequence[str], rows: Sequence[Sequence[str]], right_align_numbers: bool = False
) -> list[str]:
    """Return aligned lines: the header, then one line per row, one width per column.

    Args:
        header: The column titles.
        rows: The rows, each as long as the header.
        right_align_numbers: Align every column but the first to the right, for
            columns of counts. Otherwise every column is aligned to the left and
            trailing spaces are removed.
    """
    widths = [max(len(cell) for cell in column) for column in zip(header, *rows, strict=True)]
    lines = [
        '  '.join(
            cell.rjust(width) if right_align_numbers and position else cell.ljust(width)
            for position, (cell, width) in enumerate(zip(row, widths, strict=True))
        )
        for row in (header, *rows)
    ]
    return lines if right_align_numbers else [line.rstrip() for line in lines]


@dataclass(frozen=True, kw_only=True)
class TextTable:
    """A table attached to a log record, so a terminal handler can draw it.

    Attributes:
        header: The column titles.
        rows: The rows, each as long as the header.
        right_align_numbers: Every column but the first holds counts.
        total: The last row is a total.
    """

    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    right_align_numbers: bool = False
    total: bool = False


def log_table(
    *,
    logger: logging.Logger,
    header: Sequence[str],
    rows: Sequence[Sequence[str]],
    right_align_numbers: bool = False,
    total: bool = False,
) -> None:
    """Log a table as one record of aligned plain lines, with the table attached.

    Log files and tests get the plain lines; the command line handler draws the
    attached ``TextTable`` when it writes to a terminal.

    Args:
        logger: The logger to write to, at INFO.
        header: The column titles.
        rows: The rows, each as long as the header.
        right_align_numbers: Align every column but the first to the right.
        total: The last row is a total.
    """
    table = TextTable(
        header=tuple(header),
        rows=tuple(tuple(row) for row in rows),
        right_align_numbers=right_align_numbers,
        total=total,
    )
    lines = format_table(header=header, rows=rows, right_align_numbers=right_align_numbers)
    logger.info('%s', '\n'.join(lines), extra={'tslab_table': table}, stacklevel=2)

"""Text helpers: first lines and aligned tables, plain enough for log files."""

from collections.abc import Sequence
from dataclasses import dataclass
import logging

__all__ = ['TextTable', 'first_line', 'format_table', 'log_table']


def first_line(*, text: str) -> str:
    """Return the first line of ``text``, or ``''`` if it has none.

    Args:
        text: Any text, for example an exception message.
    """
    lines = text.splitlines()
    return lines[0] if lines else ''


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

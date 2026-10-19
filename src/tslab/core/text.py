"""Small text helpers shared by the engine and the layers above it."""

from collections.abc import Sequence

__all__ = ['describe_error', 'first_line', 'format_table', 'qualified_name']


def first_line(*, text: str) -> str:
    """Return the first line of ``text``, or ``''`` if it has none.

    Args:
        text: Any text, for example an exception message.
    """
    lines = text.splitlines()
    return lines[0] if lines else ''


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

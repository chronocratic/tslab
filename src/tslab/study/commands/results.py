"""The results command: write the results table of the runs as CSV or JSON."""

from collections.abc import Mapping, Sequence
import json
import logging
from pathlib import Path
import sys
from typing import Protocol

from tslab.core.files import atomic_write_text
from tslab.enums import OutputFormat
from tslab.study.context import Context
from tslab.study.reporting.result_table import collect_results, format_results_csv, ResultRow

__all__ = ['results']

_LOGGER = logging.getLogger(__name__)


class _Formatter(Protocol):
    def __call__(self, *, rows: Sequence[ResultRow]) -> str: ...


def _format_json(*, rows: Sequence[ResultRow]) -> str:
    return json.dumps({'schema': 'tslab.study-results/1', 'rows': rows}, indent=2)


_FORMATTERS: Mapping[OutputFormat, _Formatter] = {
    OutputFormat.TEXT: format_results_csv,
    OutputFormat.JSON: _format_json,
}


def results(*, context: Context, out: Path, output_format: OutputFormat = OutputFormat.TEXT) -> int:
    """Write the results table of the runs as CSV or JSON.

    Args:
        context: The context of the command.
        out: The output file, or ``Path('-')`` for standard output.
        output_format: ``TEXT`` for CSV, or ``JSON`` for rows in a versioned JSON document.

    Returns:
        ``0``.
    """
    rows = collect_results(
        study=context.study, study_directory=context.study_directory, reports=context.reports
    )
    text = _FORMATTERS[output_format](rows=rows)
    if out == Path('-'):
        sys.stdout.write(text if text.endswith('\n') else f'{text}\n')
    else:
        atomic_write_text(path=out, text=text)
    _LOGGER.info('wrote %d rows to %s', len(rows), out.as_posix())
    return 0

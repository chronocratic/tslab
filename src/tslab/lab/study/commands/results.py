"""The results command: write the results table of the runs as CSV or JSON."""

from collections.abc import Mapping, Sequence
from enum import StrEnum
import json
import logging
from pathlib import Path
import sys
from typing import Protocol

from tslab.lab.core.files import atomic_write_text
from tslab.lab.study.context import Context
from tslab.lab.study.reporting.result_table import collect_results, format_results_csv, ResultRow

__all__ = ['ResultsFormat', 'results']

_LOGGER = logging.getLogger(__name__)


class ResultsFormat(StrEnum):
    """The file format of the results table.

    Attributes:
        CSV: One row per run, as CSV.
        JSON: The rows in a versioned JSON document.
    """

    CSV = 'csv'
    JSON = 'json'


class _Formatter(Protocol):
    def __call__(self, *, rows: Sequence[ResultRow]) -> str: ...


def _format_json(*, rows: Sequence[ResultRow]) -> str:
    return json.dumps({'schema': 'tslab.study-results/1', 'rows': rows}, indent=2)


_FORMATTERS: Mapping[ResultsFormat, _Formatter] = {
    ResultsFormat.CSV: format_results_csv,
    ResultsFormat.JSON: _format_json,
}


def results(
    *, context: Context, out: Path, results_format: ResultsFormat = ResultsFormat.CSV
) -> int:
    """Write the results table of the runs as CSV or JSON.

    Args:
        context: The context of the command.
        out: The output file, or ``Path('-')`` for standard output.
        results_format: ``CSV``, or ``JSON`` for rows in a versioned JSON document.

    Returns:
        ``0``.
    """
    rows = collect_results(
        study=context.study, study_directory=context.study_directory, reports=context.reports
    )
    text = _FORMATTERS[results_format](rows=rows)
    if out == Path('-'):
        sys.stdout.write(text if text.endswith('\n') else f'{text}\n')
    else:
        atomic_write_text(path=out, text=text)
    _LOGGER.info('wrote %d rows to %s', len(rows), out.as_posix())
    return 0

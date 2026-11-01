"""The results command: write the results table of the runs as CSV or JSON."""

import json
import logging
from pathlib import Path
import sys

from tslab.core.files import atomic_write_text
from tslab.study.context import Context
from tslab.study.result_table import collect_results, format_results_csv, write_results_csv

__all__ = ['results']

_LOGGER = logging.getLogger(__name__)


def results(*, context: Context, out: Path, as_json: bool = False) -> int:
    """Write the results table of the runs as CSV or JSON.

    Args:
        context: The context of the command.
        out: The output file, or ``Path('-')`` for standard output.
        as_json: Write rows as a versioned JSON document.

    Returns:
        ``0``.
    """
    rows = collect_results(
        study=context.study, study_directory=context.study_directory, reports=context.reports
    )
    if out == Path('-'):
        text = (
            json.dumps({'schema': 'tslab.study-results/1', 'rows': rows}, indent=2)
            if as_json
            else format_results_csv(rows=rows)
        )
        sys.stdout.write(f'{text}\n' if as_json else text)
    elif as_json:
        atomic_write_text(
            path=out, text=json.dumps({'schema': 'tslab.study-results/1', 'rows': rows}, indent=2)
        )
    else:
        write_results_csv(rows=rows, path=out)
    _LOGGER.info('wrote %d rows to %s', len(rows), out.as_posix())
    return 0

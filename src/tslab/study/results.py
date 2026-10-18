"""The results table: one row per instance, every metric of every run.

The metrics of a run live in the ``data.json`` of the instances whose ``meta.json``
names the ``Metrics`` artifact. Malformed files must never stop the table:
they become an ``unreadable`` row with the error text.
"""

from collections.abc import Sequence
import csv
import io
import json
from pathlib import Path

from tslab.core.artifact import Metrics
from tslab.core.paths import atomic_write_text
from tslab.core.status import InstanceRecord
from tslab.study.definition import Study
from tslab.study.state import RunReport

__all__ = ['METRICS_TYPE', 'ResultRow', 'collect_results', 'write_results_csv']

#: The ``meta.json`` value of a metrics instance, ``module.Class`` of ``Metrics``.
METRICS_TYPE = f'{Metrics.__module__}.{Metrics.__qualname__}'

#: One row of the results table: column name to value.
type ResultRow = dict[str, str | float | None]


def collect_results(
    *, study: Study, study_directory: Path, reports: Sequence[RunReport]
) -> list[ResultRow]:
    """One row per (run, instance) of every report, plus one row per bare run.

    Args:
        study: The study the reports are for; its axes become columns.
        study_directory: The ``<output_root>/<study>`` directory.
        reports: The state report of every selected run.

    Returns:
        The rows, in report order, the instances in ``status.json`` order; the
        axis columns come from the grid of the run and are empty where the grid
        lacks the axis.
    """
    axes = _axis_names(study=study)
    rows: list[ResultRow] = []
    for report in reports:
        snapshot = report.folder.snapshot
        if snapshot is None or not snapshot.instances:
            rows.append(_bare_row(report=report, axes=axes))
            continue
        for instance_id, record in snapshot.instances.items():
            rows.append(
                _instance_row(
                    report=report,
                    axes=axes,
                    instance_id=instance_id,
                    record=record,
                    run_directory=study_directory / report.run.run_path,
                )
            )
    _normalize(rows=rows)
    return rows


def _axis_names(*, study: Study) -> list[str]:
    """The axis names of every grid, in first-appearance order."""
    axes: list[str] = []
    for grid in study.grids.values():
        for axis in grid.axes:
            if axis not in axes:
                axes.append(axis)
    return axes


def _row_head(*, report: RunReport, axes: Sequence[str]) -> ResultRow:
    row: ResultRow = {'grid': report.run.grid}
    for axis in axes:
        row[axis] = report.run.values.get(axis)
    row['run_path'] = report.run.run_path
    row['study_state'] = report.state.value
    return row


def _bare_row(*, report: RunReport, axes: Sequence[str]) -> ResultRow:
    row = _row_head(report=report, axes=axes)
    row['instance'] = None
    row['instance_state'] = None
    row['error'] = report.folder.read_error
    return row


def _instance_row(
    *,
    report: RunReport,
    axes: Sequence[str],
    instance_id: str,
    record: InstanceRecord,
    run_directory: Path,
) -> ResultRow:
    row = _row_head(report=report, axes=axes)
    row['instance'] = instance_id
    row['instance_state'] = record.state.value
    row['error'] = _first_line(record.error)
    row.update(
        _metrics_of(
            instance_directory=_instance_directory(
                run_directory=run_directory, instance_id=instance_id
            )
        )
    )
    return row


def _instance_directory(*, run_directory: Path, instance_id: str) -> Path:
    """The folder of one instance: ``stage[a/b]`` maps to ``stage/a/b``."""
    stage, _, name = instance_id.partition('[')
    return run_directory.joinpath(stage, *name.rstrip(']').split('/'))


def _metrics_of(*, instance_directory: Path) -> dict[str, str | float | None]:
    """The ``metric/<name>`` cells of one instance; ``{}`` for a non-metrics one.

    An unreadable or malformed ``meta.json`` or ``data.json`` gives one
    ``_unreadable`` marker cell that the row turns into an ``unreadable`` state.
    """
    try:
        meta = json.loads((instance_directory / 'meta.json').read_text(encoding='utf-8'))
    except (OSError, ValueError, UnicodeDecodeError) as error:
        return {'instance_state': 'unreadable', 'error': f'{type(error).__name__}: {error}'}
    if not isinstance(meta, dict) or meta.get('artifact_type') != METRICS_TYPE:
        return {}
    try:
        data = json.loads((instance_directory / 'data.json').read_text(encoding='utf-8'))
    except (OSError, ValueError, UnicodeDecodeError) as error:
        return {'instance_state': 'unreadable', 'error': f'{type(error).__name__}: {error}'}
    if not isinstance(data, dict):
        return {'instance_state': 'unreadable', 'error': 'data.json is not a JSON object'}
    cells: dict[str, str | float | None] = {}
    for name, value in data.items():
        cells[f'metric/{name}'] = (
            float(value) if not isinstance(value, bool) and isinstance(value, int | float) else None
        )
    return cells


def _first_line(value: str | None) -> str | None:
    return None if value is None else value.splitlines()[0]


def _normalize(*, rows: list[ResultRow]) -> None:
    """Give every row every column, empty cells as ``None``."""
    for row in rows:
        for other in rows:
            for key in other:
                row.setdefault(key, None)


def write_results_csv(*, rows: Sequence[ResultRow], path: Path) -> None:
    """Write the rows as CSV, ``None`` as an empty cell, atomically.

    Args:
        rows: The rows to write; the header is the union of their columns,
            the metric columns sorted by name.
        path: The CSV file to write.
    """
    header = _header_of(rows=rows)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator='\n')
    writer.writerow(header)
    for row in rows:
        writer.writerow(['' if row.get(column) is None else row.get(column) for column in header])
    atomic_write_text(path=path, text=buffer.getvalue())


def _header_of(*, rows: Sequence[ResultRow]) -> list[str]:
    header: list[str] = []
    for row in rows:
        for key in row:
            if not key.startswith('metric/') and key not in header:
                header.append(key)
    header.extend(sorted(key for row in rows for key in row if key.startswith('metric/')))
    return header

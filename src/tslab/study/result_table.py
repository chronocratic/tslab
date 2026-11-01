"""The results table: one row per instance, every metric of every run.

The metrics of a run live in the ``data.json`` of the done instances whose
``meta.json`` names the ``Metrics`` artifact. Malformed files must never stop
the table: they become an ``unreadable`` row with the error text.
"""

from collections.abc import Sequence
import csv
import io
import json
from pathlib import Path

from tslab.core.artifact import Metrics
from tslab.core.paths import atomic_write_text, META_FILE, require_path
from tslab.core.status import InstanceRecord, InstanceState
from tslab.core.text import first_line
from tslab.study.context import context_for, Selection
from tslab.study.definition import Study
from tslab.study.state import RunReport

__all__ = [
    'METRICS_TYPE',
    'ResultRow',
    'collect_results',
    'format_results_csv',
    'instance_directory',
    'study_results',
    'write_results_csv',
]

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


def study_results(*, study: Study, machine: Path) -> list[ResultRow]:
    """The rows of the results table of every run of a study.

    Args:
        study: The study to read the results of.
        machine: The machine profile the runs run on.

    Returns:
        One row per (run, instance), and one row per run without
        instances; the runs of every grid, in expansion order.
    """
    context = context_for(
        study=study,
        machine_path=machine,
        selection=Selection(),
        project_root=Path.cwd(),
        read_only=True,
    )
    return collect_results(
        study=study, study_directory=context.study_directory, reports=context.reports
    )


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
    row['instance_state'] = None if report.folder.read_error is None else 'unreadable'
    row['seconds'] = None
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
    row['seconds'] = record.seconds
    row['error'] = None if record.error is None else first_line(text=record.error)
    # why: only a done instance has a meta.json; the others show their own state and error
    if record.state is InstanceState.DONE:
        row.update(
            _metrics_of(
                instance_directory=instance_directory(
                    run_directory=run_directory, instance_id=instance_id
                )
            )
        )
    return row


def instance_directory(*, run_directory: Path, instance_id: str) -> Path:
    """The folder of one instance: ``stage[a/b]`` maps to ``stage/a/b``.

    Args:
        run_directory: The folder of the run.
        instance_id: The instance name from ``status.json``.

    Returns:
        The folder that holds the instance artifacts.
    """
    require_path(value=run_directory, parameter='run_directory')
    stage, _, name = instance_id.partition('[')
    return run_directory.joinpath(stage, *name.rstrip(']').split('/'))


def _metrics_of(*, instance_directory: Path) -> dict[str, str | float | None]:
    """The ``metric/<name>`` cells of one instance; ``{}`` for a non-metrics one.

    An unreadable or malformed ``meta.json`` or ``data.json`` gives one
    ``_unreadable`` marker cell that the row turns into an ``unreadable`` state.
    """
    try:
        meta = json.loads((instance_directory / META_FILE).read_text(encoding='utf-8'))
    except (OSError, ValueError, UnicodeDecodeError) as error:
        return {'instance_state': 'unreadable', 'error': f'{type(error).__name__}: {error}'}
    if not isinstance(meta, dict) or meta.get('artifact_type') != METRICS_TYPE:
        return {}
    try:
        data = json.loads((instance_directory / Metrics.FILE_NAME).read_text(encoding='utf-8'))
    except (OSError, ValueError, UnicodeDecodeError) as error:
        return {'instance_state': 'unreadable', 'error': f'{type(error).__name__}: {error}'}
    if not isinstance(data, dict):
        return {
            'instance_state': 'unreadable',
            'error': f'{Metrics.FILE_NAME} is not a JSON object',
        }
    cells: dict[str, str | float | None] = {}
    for name, value in data.items():
        cells[f'metric/{name}'] = (
            float(value) if not isinstance(value, bool) and isinstance(value, int | float) else None
        )
    return cells


def _normalize(*, rows: list[ResultRow]) -> None:
    """Give every row every column, empty cells as ``None``."""
    keys = {key: None for row in rows for key in row}
    for row in rows:
        for key in keys:
            row.setdefault(key, None)


def write_results_csv(*, rows: Sequence[ResultRow], path: Path) -> None:
    """Write the rows as CSV, ``None`` as an empty cell, atomically.

    Args:
        rows: The rows to write; the header is the union of their columns,
            the metric columns sorted by name.
        path: The CSV file to write.
    """
    atomic_write_text(path=path, text=format_results_csv(rows=rows))


def format_results_csv(*, rows: Sequence[ResultRow]) -> str:
    """Format result rows as CSV text.

    Args:
        rows: The rows to format; the header is their union of columns.

    Returns:
        CSV text with a trailing newline.
    """
    header = _header_of(rows=rows)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator='\n')
    writer.writerow(header)
    for row in rows:
        writer.writerow(['' if row.get(column) is None else row.get(column) for column in header])
    return buffer.getvalue()


def _header_of(*, rows: Sequence[ResultRow]) -> list[str]:
    keys = {key: None for row in rows for key in row}
    header = [key for key in keys if not key.startswith('metric/')]
    return [*header, *sorted(key for key in keys if key.startswith('metric/'))]

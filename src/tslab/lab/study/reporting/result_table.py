"""The results table: one row per instance, every metric of every run.

The metrics of a run live in the ``data.json`` of the done instances whose
``meta.json`` names the ``Metrics`` artifact. Malformed files must never stop
the table: they become an ``unreadable`` row with the error text.
"""

from collections.abc import Mapping, Sequence
import csv
import io
import json
from pathlib import Path

from tslab.lab.core.artifact import Metrics
from tslab.lab.core.files import atomic_write_text
from tslab.lab.core.paths import META_FILE
from tslab.lab.core.status import InstanceRecord
from tslab.lab.core.text import first_line
from tslab.lab.enums import InstanceState
from tslab.lab.study.context import context_for, Selection
from tslab.lab.study.definition.columns import (
    ERROR_COLUMN,
    GRID_COLUMN,
    INSTANCE_COLUMN,
    INSTANCE_STATE_COLUMN,
    RUN_PATH_COLUMN,
    SECONDS_COLUMN,
    STUDY_STATE_COLUMN,
)
from tslab.lab.study.definition.definition import Study
from tslab.lab.study.launchers.cpus import available_cpus
from tslab.lab.study.paths import instance_directory
from tslab.lab.study.reporting.state import RunReport

__all__ = [
    'METRICS_TYPE',
    'METRIC_PREFIX',
    'UNREADABLE_STATE',
    'ResultRow',
    'collect_results',
    'format_results_csv',
    'study_results',
    'write_results_csv',
]

#: The ``meta.json`` value of a metrics instance, ``module.Class`` of ``Metrics``.
METRICS_TYPE = f'{Metrics.__module__}.{Metrics.__qualname__}'

#: The prefix of a metric column; the rest is the metric name.
METRIC_PREFIX = 'metric/'
#: The ``instance_state`` of an instance whose files cannot be read.
UNREADABLE_STATE = 'unreadable'

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


def study_results(
    *, study: Study, machine: Path, environment: Mapping[str, str]
) -> list[ResultRow]:
    """The rows of the results table of every run of a study.

    Args:
        study: The study to read the results of.
        machine: The machine profile the runs run on.
        environment: The process environment used by the machine profile.

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
        local_cpus=available_cpus(),
        environment=environment,
    )
    return collect_results(
        study=study, study_directory=context.study_directory, reports=context.reports
    )


def _axis_names(*, study: Study) -> list[str]:
    """The axis names of every grid, in first-appearance order."""
    return list(dict.fromkeys(axis for grid in study.grids.values() for axis in grid.axes))


def _row_head(*, report: RunReport, axes: Sequence[str]) -> ResultRow:
    row: ResultRow = {GRID_COLUMN: report.run.grid}
    for axis in axes:
        row[axis] = report.run.values.get(axis)
    row[RUN_PATH_COLUMN] = report.run.run_path.as_posix()
    row[STUDY_STATE_COLUMN] = report.state.value
    return row


def _bare_row(*, report: RunReport, axes: Sequence[str]) -> ResultRow:
    row = _row_head(report=report, axes=axes)
    row[INSTANCE_COLUMN] = None
    row[INSTANCE_STATE_COLUMN] = None if report.folder.read_error is None else UNREADABLE_STATE
    row[SECONDS_COLUMN] = None
    row[ERROR_COLUMN] = report.folder.read_error
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
    row[INSTANCE_COLUMN] = instance_id
    row[INSTANCE_STATE_COLUMN] = record.state.value
    row[SECONDS_COLUMN] = record.seconds
    row[ERROR_COLUMN] = None if record.error is None else first_line(text=record.error)
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


def _unreadable_cells(*, error: str) -> dict[str, str | float | None]:
    """The cells that turn a row into an ``unreadable`` one with the error text."""
    return {INSTANCE_STATE_COLUMN: UNREADABLE_STATE, ERROR_COLUMN: error}


def _metrics_of(*, instance_directory: Path) -> dict[str, str | float | None]:
    """The ``metric/<name>`` cells of one instance; ``{}`` for a non-metrics one.

    An unreadable or malformed ``meta.json`` or ``data.json`` gives one
    ``_unreadable`` marker cell that the row turns into an ``unreadable`` state.
    """
    try:
        meta = json.loads((instance_directory / META_FILE).read_text(encoding='utf-8'))
    except (OSError, ValueError, UnicodeDecodeError) as error:
        return _unreadable_cells(error=f'{type(error).__name__}: {error}')
    if not isinstance(meta, dict) or meta.get('artifact_type') != METRICS_TYPE:
        return {}
    try:
        data = json.loads((instance_directory / Metrics.FILE_NAME).read_text(encoding='utf-8'))
    except (OSError, ValueError, UnicodeDecodeError) as error:
        return _unreadable_cells(error=f'{type(error).__name__}: {error}')
    if not isinstance(data, dict):
        return _unreadable_cells(error=f'{Metrics.FILE_NAME} is not a JSON object')
    cells: dict[str, str | float | None] = {}
    for name, value in data.items():
        cells[f'{METRIC_PREFIX}{name}'] = (
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
    header = [key for key in keys if not key.startswith(METRIC_PREFIX)]
    return [*header, *sorted(key for key in keys if key.startswith(METRIC_PREFIX))]

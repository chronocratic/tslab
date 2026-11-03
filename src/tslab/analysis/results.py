"""Study results as a frame: loading, coverage, baseline deltas and method matrices.

The frame is the results table of the study (one row per run and instance) with the
axis columns of the grids and ``metric/<name>`` columns. One run has several metric
rows (the clean evaluation and one per noise level), so select the ``instance`` column
before any comparison.
"""

from collections.abc import Mapping, Sequence
import contextlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from tslab.analysis.stats import Direction
from tslab.core.paths import require_path
from tslab.enums import InstanceState
from tslab.study.definition import Study
from tslab.study.reporting.result_table import (
    INSTANCE_COLUMN,
    INSTANCE_STATE_COLUMN,
    METRIC_PREFIX,
    SECONDS_COLUMN,
)

__all__ = ['AnalysisMetric', 'against_baseline', 'load_results', 'method_matrix', 'missing_runs']

BASELINE_COLUMN = 'baseline_value'
DELTA_COLUMN = 'delta'


@dataclass(frozen=True, kw_only=True)
class AnalysisMetric:
    """A metric to analyse: which results column, how to label it, which direction is better.

    Attributes:
        name: The metric name as the study writes it (the part after ``metric/``).
        label: Text for tables and figures.
        direction: Whether a larger value is better.
    """

    name: str
    label: str
    direction: Direction

    @property
    def column(self) -> str:
        """The name of the results column that holds this metric."""
        return f'{METRIC_PREFIX}{self.name}'


def load_results(
    *, study: Study, machine: Path, project_root: Path, environment: Mapping[str, str]
) -> pd.DataFrame:
    """Read the results table of a study into a frame.

    The machine profile and its relative ``output_root`` are read against the working
    folder, so the read runs inside ``project_root``, the folder the command line would
    run from. A notebook usually starts in ``notebooks/``, which is why it is needed.

    Args:
        study: The study to read the results of.
        machine: The machine profile, relative to ``project_root`` or absolute.
        project_root: The project folder.
        environment: The process environment used by the machine profile.

    Returns:
        One row per run and instance; metric columns are ``metric/<name>``.
    """
    require_path(value=machine, parameter='machine')
    require_path(value=project_root, parameter='project_root')
    # why: the working folder is process-wide, so this is not safe to call from two
    # threads at once; a notebook never does
    with contextlib.chdir(project_root):
        rows = study.results(machine=machine, environment=environment)
    return pd.DataFrame(rows)


def missing_runs(*, results: pd.DataFrame) -> pd.DataFrame:
    """Return the rows whose instance is not done, and runs without instances.

    Args:
        results: The frame from :func:`load_results`.

    Returns:
        The rows without a finished instance, without the seconds and metric columns.
    """
    unfinished = results[results[INSTANCE_STATE_COLUMN] != InstanceState.DONE.value]
    metrics = [column for column in unfinished.columns if column.startswith(METRIC_PREFIX)]
    return unfinished.drop(columns=[SECONDS_COLUMN, *metrics], errors='ignore')


def _require_unique(*, frame: pd.DataFrame, keys: Sequence[str], hint: str) -> None:
    duplicated = frame[frame.duplicated(subset=list(keys), keep=False)]
    if not duplicated.empty:
        first = duplicated.iloc[0][list(keys)].to_dict()
        message = f'more than one row for {first}; {hint}'
        raise ValueError(message)


def against_baseline(
    *,
    frame: pd.DataFrame,
    condition: str,
    baseline: str,
    keys: Sequence[str],
    metric: AnalysisMetric,
) -> pd.DataFrame:
    """Compare every non-baseline row with the baseline row that has the same keys.

    Args:
        frame: Rows with the ``condition`` column and ``metric.column``.
        condition: The column that tells the baseline from the other conditions.
        baseline: The value of ``condition`` that marks the baseline rows.
        keys: Columns that identify a unit (for example dataset, method, seed).
        metric: The metric, whose direction sets the sign of the delta.

    Returns:
        The non-baseline rows with ``baseline_value`` and ``delta``; a negative
        delta always means worse than the baseline.

    Raises:
        ValueError: If two baseline rows share the same ``keys``.
    """
    is_baseline = frame[condition] == baseline
    reference = frame.loc[is_baseline, [*keys, metric.column]].rename(
        columns={metric.column: BASELINE_COLUMN}
    )
    _require_unique(
        frame=reference, keys=keys, hint=f'the baseline needs unique keys; add to {list(keys)!r}'
    )
    merged = frame[~is_baseline].merge(reference, on=list(keys), how='inner')
    delta = (merged[metric.column] - merged[BASELINE_COLUMN]) * metric.direction.sign
    return merged.assign(**{DELTA_COLUMN: delta})


def method_matrix(
    *, frame: pd.DataFrame, unit: str, method: str, value: str, average_over: Sequence[str]
) -> pd.DataFrame:
    """Pivot scores to a matrix of rows by unit and columns by method.

    Args:
        frame: One row per unit, method and ``average_over`` combination.
        unit: Column for the rows (for example the dataset).
        method: Column for the columns (for example the recipe).
        value: The score column.
        average_over: Columns averaged away (for example the seed).

    Returns:
        The mean score for each unit and method.

    Raises:
        ValueError: If two rows share the unit, method and ``average_over`` values.
    """
    _require_unique(
        frame=frame,
        keys=[unit, method, *average_over],
        hint=f'filter on the {INSTANCE_COLUMN!r} column first, so scores are not mixed',
    )
    return frame.pivot_table(index=unit, columns=method, values=value, aggfunc='mean')

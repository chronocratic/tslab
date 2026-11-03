"""Paired tests, multiple-comparison correction and effect sizes for method comparisons.

Every function takes keyword arguments and returns a new object. Missing cells are
never filled: a test uses the pairs both methods have.
"""

from collections.abc import Mapping, Sequence
import dataclasses
from dataclasses import dataclass
from enum import StrEnum
import itertools

import numpy as np
import pandas as pd
import scipy.stats

__all__ = [
    'DEFAULT_ALPHA',
    'DEFAULT_BOOTSTRAP_SAMPLES',
    'DEFAULT_CONFIDENCE',
    'DEFAULT_MIN_PAIRS',
    'MIN_FRIEDMAN_DATASETS',
    'MIN_FRIEDMAN_METHODS',
    'Direction',
    'FriedmanTest',
    'PairedTest',
    'annotate_tests',
    'average_ranks',
    'bootstrap_median',
    'correlations',
    'friedman',
    'holm_adjust',
    'pairwise_wilcoxon',
    'per_unit_ranks',
    'rank_agreement',
    'rank_biserial',
    'stars',
    'wilcoxon_paired',
]

DEFAULT_ALPHA = 0.05
DEFAULT_MIN_PAIRS = 3
DEFAULT_BOOTSTRAP_SAMPLES = 2000
DEFAULT_CONFIDENCE = 0.95
MIN_FRIEDMAN_METHODS = 3
MIN_FRIEDMAN_DATASETS = 2
NOT_SIGNIFICANT = 'n.s.'
NOT_AVAILABLE = 'n/a'
# A p-value strictly below alpha / divisor earns the marker, strongest first.
_STAR_LEVELS = ((50, '***'), (5, '**'), (1, '*'))
_PAIR_COLUMNS = ('method_a', 'method_b', 'n', 'median_diff', 'statistic', 'p_value', 'effect_size')


class Direction(StrEnum):
    """Whether a larger metric value is better or worse."""

    HIGHER_IS_BETTER = 'higher_is_better'
    LOWER_IS_BETTER = 'lower_is_better'

    @property
    def sign(self) -> int:
        """Return +1 when larger is better and -1 when smaller is better."""
        return _SIGNS[self]


_SIGNS: Mapping[Direction, int] = {Direction.HIGHER_IS_BETTER: 1, Direction.LOWER_IS_BETTER: -1}


@dataclass(frozen=True, kw_only=True)
class PairedTest:
    """The outcome of a paired Wilcoxon signed-rank test.

    Attributes:
        n: Number of complete pairs.
        median_diff: Median of left minus right.
        statistic: The Wilcoxon statistic, NaN when the test was skipped.
        p_value: The two-sided p-value, NaN when the test was skipped.
        effect_size: Matched-pairs rank-biserial correlation in [-1, 1].
    """

    n: int
    median_diff: float
    statistic: float
    p_value: float
    effect_size: float


def rank_biserial(*, differences: np.ndarray) -> float:
    """Return the matched-pairs rank-biserial correlation of paired differences.

    Args:
        differences: Left minus right for each pair. Zeros are ignored.

    Returns:
        ``(W+ - W-) / (W+ + W-)``, or 0.0 when no non-zero difference is left.
    """
    nonzero = differences[differences != 0]
    if nonzero.size == 0:
        return 0.0
    ranks = scipy.stats.rankdata(np.abs(nonzero))
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / (positive + negative)


def wilcoxon_paired(
    *, left: pd.Series, right: pd.Series, min_pairs: int = DEFAULT_MIN_PAIRS
) -> PairedTest:
    """Run a two-sided paired Wilcoxon signed-rank test on the shared, complete pairs.

    Args:
        left: Scores of the first method, indexed by unit.
        right: Scores of the second method, indexed by unit.
        min_pairs: Fewer complete pairs than this skip the test.

    Returns:
        The test. When it is skipped (too few pairs, or every difference is zero)
        the statistic and p-value are NaN and the effect size is 0.0.
    """
    pairs = pd.concat([left, right], axis=1, join='inner').dropna()
    differences = (pairs.iloc[:, 0] - pairs.iloc[:, 1]).to_numpy(dtype=float)
    median_diff = float(np.median(differences)) if differences.size else float('nan')
    if differences.size < min_pairs or not differences.any():
        return PairedTest(
            n=int(differences.size),
            median_diff=median_diff,
            statistic=float('nan'),
            p_value=float('nan'),
            effect_size=0.0,
        )
    result = scipy.stats.wilcoxon(differences)
    return PairedTest(
        n=int(differences.size),
        median_diff=median_diff,
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        effect_size=rank_biserial(differences=differences),
    )


def holm_adjust(*, p_values: pd.Series) -> pd.Series:
    """Adjust p-values with the Holm step-down method.

    Args:
        p_values: Raw p-values. NaN entries stay NaN and are not counted.

    Returns:
        Adjusted p-values with the same index and order, capped at 1.
    """
    values = p_values.to_numpy(dtype=float)
    valid = np.flatnonzero(~np.isnan(values))
    order = valid[np.argsort(values[valid], kind='stable')]
    count = len(order)
    adjusted = np.full(len(values), np.nan)
    running = 0.0
    for position, index in enumerate(order):
        running = max(running, (count - position) * values[index])
        adjusted[index] = min(running, 1.0)
    return pd.Series(adjusted, index=p_values.index, dtype=float)


def stars(*, p_value: float, alpha: float = DEFAULT_ALPHA) -> str:
    """Return the significance marker for a p-value.

    Args:
        p_value: The (usually adjusted) p-value.
        alpha: Significance level; ``*`` below alpha, ``**`` below alpha / 5,
            ``***`` below alpha / 50.

    Returns:
        ``***``, ``**``, ``*``, ``n.s.``, or ``n/a`` for NaN.
    """
    if np.isnan(p_value):
        return NOT_AVAILABLE
    return next(
        (marker for divisor, marker in _STAR_LEVELS if p_value < alpha / divisor), NOT_SIGNIFICANT
    )


def annotate_tests(*, table: pd.DataFrame, alpha: float = DEFAULT_ALPHA) -> pd.DataFrame:
    """Add Holm-adjusted p-values and significance markers to a table of tests.

    Args:
        table: A frame with a ``p_value`` column.
        alpha: Significance level for the markers.

    Returns:
        A new frame with the extra columns ``p_holm`` and ``significance``.
    """
    adjusted = holm_adjust(p_values=table['p_value'])
    markers = [stars(p_value=float(value), alpha=alpha) for value in adjusted]
    return table.assign(p_holm=adjusted, significance=pd.Series(markers, index=table.index))


def pairwise_wilcoxon(
    *, matrix: pd.DataFrame, alpha: float = DEFAULT_ALPHA, min_pairs: int = DEFAULT_MIN_PAIRS
) -> pd.DataFrame:
    """Test every pair of method columns with a paired Wilcoxon test.

    Args:
        matrix: Rows by unit (for example dataset), columns by method.
        alpha: Significance level for the markers.
        min_pairs: Fewer complete pairs than this skip a test.

    Returns:
        One row per column pair, with Holm-adjusted p-values and markers.
    """
    rows = [
        {
            'method_a': method_a,
            'method_b': method_b,
            **dataclasses.asdict(
                wilcoxon_paired(left=matrix[method_a], right=matrix[method_b], min_pairs=min_pairs)
            ),
        }
        for method_a, method_b in itertools.combinations(matrix.columns, 2)
    ]
    return annotate_tests(table=pd.DataFrame(rows, columns=list(_PAIR_COLUMNS)), alpha=alpha)


@dataclass(frozen=True, kw_only=True)
class FriedmanTest:
    """The outcome of a Friedman test over complete rows.

    Attributes:
        statistic: The chi-square statistic, NaN when the test was skipped.
        p_value: The p-value, NaN when the test was skipped.
        datasets: Number of complete rows used.
        methods: Number of method columns.
    """

    statistic: float
    p_value: float
    datasets: int
    methods: int


def friedman(*, matrix: pd.DataFrame) -> FriedmanTest:
    """Run a Friedman test across method columns, using only complete rows.

    Args:
        matrix: Rows by unit (for example dataset), columns by method.

    Returns:
        The test. With fewer than 3 methods, fewer than 2 complete rows, or every
        score tied in every row, the statistic and p-value are NaN.
    """
    complete = matrix.dropna()
    methods = matrix.shape[1]
    all_tied = bool((complete.max(axis=1) == complete.min(axis=1)).all())
    if methods < MIN_FRIEDMAN_METHODS or len(complete) < MIN_FRIEDMAN_DATASETS or all_tied:
        return FriedmanTest(
            statistic=float('nan'), p_value=float('nan'), datasets=len(complete), methods=methods
        )
    result = scipy.stats.friedmanchisquare(*(complete[column] for column in complete.columns))
    return FriedmanTest(
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        datasets=len(complete),
        methods=methods,
    )


def bootstrap_median(
    *,
    values: pd.Series | np.ndarray,
    rng: np.random.Generator,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
) -> tuple[float, float, float]:
    """Return the median and a percentile bootstrap confidence interval.

    Args:
        values: The observations. NaN entries are dropped.
        rng: Random generator, so a seed makes the result repeatable.
        samples: Number of resamples.
        confidence: Coverage of the interval.

    Returns:
        ``(median, lower, upper)``; all NaN when there are no observations.
    """
    data = np.asarray(values, dtype=float)
    data = data[~np.isnan(data)]
    if data.size == 0:
        nan = float('nan')
        return nan, nan, nan
    medians = np.median(rng.choice(data, size=(samples, data.size), replace=True), axis=1)
    tail = (1 - confidence) / 2 * 100
    lower, upper = np.percentile(medians, [tail, 100 - tail])
    return float(np.median(data)), float(lower), float(upper)


def rank_agreement(*, left: pd.DataFrame, right: pd.DataFrame) -> pd.Series:
    """Return the Kendall tau between two score tables for each shared row.

    Args:
        left: Rows by unit, columns by method.
        right: Same layout; only shared rows and columns are compared.

    Returns:
        Kendall tau per shared row. Rows with a missing value are dropped.
    """
    columns = left.columns.intersection(right.columns)
    rows = left.index.intersection(right.index)
    first = left.loc[rows, columns].dropna()
    second = right.loc[first.index, columns].dropna()
    first = first.loc[second.index]
    taus = {
        row: float(scipy.stats.kendalltau(first.loc[row], second.loc[row]).statistic)
        for row in first.index
    }
    return pd.Series(taus, index=first.index, dtype=float)


def correlations(
    *,
    frame: pd.DataFrame,
    columns: Sequence[str],
    target: str,
    alpha: float = DEFAULT_ALPHA,
    min_pairs: int = DEFAULT_MIN_PAIRS,
) -> pd.DataFrame:
    """Return the Spearman correlation of each column with a target column.

    Args:
        frame: The observations.
        columns: Columns to correlate with ``target``.
        target: The column they are compared against.
        alpha: Significance level for the markers.
        min_pairs: Fewer complete pairs (or a constant column) give NaN.

    Returns:
        A frame with columns ``column, n, rho, p_value, p_holm, significance``.
    """
    rows = []
    for column in columns:
        pairs = frame[[column, target]].dropna()
        usable = len(pairs) >= min_pairs and (pairs.min() < pairs.max()).all()
        result = scipy.stats.spearmanr(pairs[column], pairs[target]) if usable else None
        rows.append(
            {
                'column': column,
                'n': len(pairs),
                'rho': float(result.statistic) if result else float('nan'),
                'p_value': float(result.pvalue) if result else float('nan'),
            }
        )
    table = pd.DataFrame(rows, columns=['column', 'n', 'rho', 'p_value'])
    return annotate_tests(table=table, alpha=alpha)


def per_unit_ranks(*, matrix: pd.DataFrame, direction: Direction) -> pd.DataFrame:
    """Rank the methods within each complete row; 1 is best and ties share the average.

    Args:
        matrix: Rows by unit, columns by method.
        direction: Which end of the scale is best.

    Returns:
        Ranks for the complete rows only, same columns.
    """
    complete = matrix.dropna()
    ranks = scipy.stats.rankdata(-direction.sign * complete.to_numpy(dtype=float), axis=1)
    return pd.DataFrame(ranks, index=complete.index, columns=complete.columns)


def average_ranks(*, matrix: pd.DataFrame, direction: Direction) -> pd.Series:
    """Return the mean rank of each method over complete rows, best first.

    Args:
        matrix: Rows by unit, columns by method.
        direction: Which end of the scale is best.

    Returns:
        Mean ranks sorted ascending (the best method first).
    """
    return per_unit_ranks(matrix=matrix, direction=direction).mean().sort_values(kind='stable')

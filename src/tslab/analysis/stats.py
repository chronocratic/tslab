"""Paired tests, multiple-comparison correction and effect sizes for method comparisons.

Every function takes keyword arguments and returns a new object. Missing cells are
never filled: a test uses the pairs both methods have.
"""

from collections.abc import Mapping
import dataclasses
from dataclasses import dataclass
from enum import StrEnum
import itertools

import numpy as np
import pandas as pd
import scipy.stats

__all__ = [
    'DEFAULT_ALPHA',
    'DEFAULT_MIN_PAIRS',
    'Direction',
    'PairedTest',
    'annotate_tests',
    'holm_adjust',
    'pairwise_wilcoxon',
    'rank_biserial',
    'stars',
    'wilcoxon_paired',
]

DEFAULT_ALPHA = 0.05
DEFAULT_MIN_PAIRS = 3
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
    valid = p_values.dropna().sort_values()
    count = len(valid)
    running = 0.0
    adjusted: dict[object, float] = {}
    for position, (label, value) in enumerate(valid.items()):
        running = max(running, (count - position) * float(value))
        adjusted[label] = min(running, 1.0)
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

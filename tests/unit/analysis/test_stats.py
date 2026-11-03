import numpy as np
import pandas as pd
import pytest
import scipy.stats

from tslab.analysis.stats import (
    annotate_tests,
    Direction,
    holm_adjust,
    pairwise_wilcoxon,
    rank_biserial,
    stars,
    wilcoxon_paired,
)


def test_direction_sign() -> None:
    assert Direction.HIGHER_IS_BETTER.sign == 1
    assert Direction.LOWER_IS_BETTER.sign == -1


@pytest.mark.parametrize(
    ('differences', 'expected'),
    [
        ([1.0, 2.0, 3.0], 1.0),
        ([1.0, 2.0, -3.0], 0.0),
        ([-1.0, 2.0, 3.0], 4 / 6),
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
    ],
)
def test_rank_biserial(differences: list[float], expected: float) -> None:
    assert rank_biserial(differences=np.array(differences)) == pytest.approx(expected)


def test_holm_adjust_keeps_order_and_nan() -> None:
    p_values = pd.Series([0.01, 0.04, 0.03, np.nan], index=list('abcd'))
    result = holm_adjust(p_values=p_values)
    assert list(result.index) == list('abcd')
    assert result.iloc[:3].tolist() == pytest.approx([0.03, 0.06, 0.06])
    assert np.isnan(result['d'])
    assert holm_adjust(p_values=pd.Series([0.6, 0.7])).max() <= 1.0


@pytest.mark.parametrize(
    ('p_value', 'marker'),
    [(0.0005, '***'), (0.005, '**'), (0.03, '*'), (0.05, 'n.s.'), (np.nan, 'n/a')],
)
def test_stars(p_value: float, marker: str) -> None:
    assert stars(p_value=p_value) == marker


def test_wilcoxon_paired_matches_scipy() -> None:
    rng = np.random.default_rng(0)
    left = pd.Series(rng.normal(size=12))
    right = left - rng.normal(0.5, 0.2, size=12)
    result = wilcoxon_paired(left=left, right=right)
    reference = scipy.stats.wilcoxon(left - right)
    assert (result.statistic, result.p_value) == pytest.approx(
        (reference.statistic, reference.pvalue)
    )
    assert result.n == 12
    assert result.median_diff == pytest.approx(float(np.median(left - right)))


def test_wilcoxon_paired_skips_few_pairs_and_all_zero() -> None:
    few = wilcoxon_paired(left=pd.Series([1.0, 2.0]), right=pd.Series([0.0, 1.0]))
    zero = wilcoxon_paired(left=pd.Series([1.0] * 5), right=pd.Series([1.0] * 5))
    for result in (few, zero):
        assert np.isnan(result.statistic)
        assert np.isnan(result.p_value)
        assert result.effect_size == 0.0


def test_wilcoxon_paired_pairs_on_shared_index_without_nan() -> None:
    left = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0], index=list('abcde'))
    right = pd.Series([0.0, 1.0, 1.0, 3.0], index=list('abcd'))
    assert wilcoxon_paired(left=left, right=right).n == 3


def test_annotate_tests_returns_a_new_frame() -> None:
    table = pd.DataFrame({'p_value': [0.01, 0.2]})
    result = annotate_tests(table=table)
    assert list(table.columns) == ['p_value']
    assert result['significance'].tolist() == ['*', 'n.s.']


def test_pairwise_wilcoxon_rows() -> None:
    matrix = pd.DataFrame(np.random.default_rng(1).normal(size=(8, 4)), columns=list('wxyz'))
    table = pairwise_wilcoxon(matrix=matrix)
    assert len(table) == 6
    assert list(table.columns) == [
        'method_a',
        'method_b',
        'n',
        'median_diff',
        'statistic',
        'p_value',
        'effect_size',
        'p_holm',
        'significance',
    ]
    assert (table['method_a'].iloc[0], table['method_b'].iloc[0]) == ('w', 'x')

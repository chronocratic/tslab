import numpy as np
import pandas as pd
import pytest
import scipy.stats

from tslab.analysis.stats import (
    annotate_tests,
    average_ranks,
    bootstrap_median,
    complete_row_count,
    correlations,
    Direction,
    friedman,
    holm_adjust,
    pairwise_wilcoxon,
    per_unit_ranks,
    rank_agreement,
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


def test_friedman_uses_complete_rows() -> None:
    matrix = pd.DataFrame(
        {'a': [1, 2, 3, 4, np.nan], 'b': [2, 3, 4, 5, 1], 'c': [3, 4, 5, 6, 1]}, dtype=float
    )
    result = friedman(matrix=matrix)
    assert (result.datasets, result.methods) == (4, 3)
    assert result.p_value < 0.05


def test_friedman_nan_when_too_small() -> None:
    assert np.isnan(friedman(matrix=pd.DataFrame({'a': [1.0, 2.0], 'b': [2.0, 3.0]})).p_value)
    assert np.isnan(friedman(matrix=pd.DataFrame({'a': [1.0], 'b': [2.0], 'c': [3.0]})).p_value)


def test_bootstrap_median_repeats_and_handles_empty() -> None:
    values = np.arange(20.0)
    first = bootstrap_median(values=values, rng=np.random.default_rng(3))
    assert first == bootstrap_median(values=values, rng=np.random.default_rng(3))
    assert first[1] <= first[0] <= first[2]
    assert first[0] == 9.5
    empty = bootstrap_median(values=np.array([np.nan]), rng=np.random.default_rng(0))
    assert all(np.isnan(empty))


def test_average_ranks_with_ties() -> None:
    matrix = pd.DataFrame({'a': [0.9, 0.8], 'b': [0.9, 0.7], 'c': [0.1, 0.1]})
    higher = average_ranks(matrix=matrix, direction=Direction.HIGHER_IS_BETTER)
    assert higher.to_dict() == {'a': 1.25, 'b': 1.75, 'c': 3.0}
    assert list(higher.index) == ['a', 'b', 'c']
    lower = average_ranks(matrix=matrix, direction=Direction.LOWER_IS_BETTER)
    assert lower.index[0] == 'c'


def test_per_unit_ranks_skip_incomplete_rows() -> None:
    matrix = pd.DataFrame({'a': [1.0, np.nan], 'b': [2.0, 1.0]}, index=['d1', 'd2'])
    ranks = per_unit_ranks(matrix=matrix, direction=Direction.HIGHER_IS_BETTER)
    assert list(ranks.index) == ['d1']
    assert ranks.loc['d1'].to_dict() == {'a': 2.0, 'b': 1.0}


def test_rank_agreement_identical_and_reversed() -> None:
    left = pd.DataFrame({'a': [1.0, 1.0], 'b': [2.0, 2.0], 'c': [3.0, 3.0]}, index=['d1', 'd2'])
    assert rank_agreement(left=left, right=left).tolist() == [1.0, 1.0]
    assert rank_agreement(left=left, right=-left).tolist() == [-1.0, -1.0]


def test_correlations_columns_and_holm() -> None:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({'x': np.arange(10.0), 'y': rng.normal(size=10), 't': np.arange(10.0)})
    table = correlations(frame=frame, columns=['x', 'y'], target='t')
    assert table['column'].tolist() == ['x', 'y']
    assert table.loc[0, 'rho'] == pytest.approx(1.0)
    assert (table['p_holm'] >= table['p_value']).all()


def test_holm_adjust_with_duplicate_index_labels() -> None:
    p_values = pd.Series([0.01, 0.04, 0.03, 0.5], index=[0, 1, 0, 1])
    assert holm_adjust(p_values=p_values).tolist() == pytest.approx([0.04, 0.09, 0.09, 0.5])


def test_correlations_rejects_target_in_columns() -> None:
    frame = pd.DataFrame({'x': [1.0, 2.0, 3.0], 't': [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="'t'"):
        correlations(frame=frame, columns=['x', 't'], target='t')


def test_complete_row_count() -> None:
    matrix = pd.DataFrame({'a': [1.0, np.nan, 3.0], 'b': [1.0, 2.0, 3.0]})
    assert complete_row_count(matrix=matrix) == 2

"""Post-hoc analysis of study results: tests, diagrams, numbers and an organised output tree.

Needs the ``analysis`` extra.
"""

from tslab.analysis.results import (
    against_baseline,
    AnalysisMetric,
    load_results,
    method_matrix,
    missing_runs,
)
from tslab.analysis.stats import (
    annotate_tests,
    DEFAULT_ALPHA,
    DEFAULT_MIN_PAIRS,
    Direction,
    holm_adjust,
    PairedTest,
    pairwise_wilcoxon,
    rank_biserial,
    stars,
    wilcoxon_paired,
)

__all__ = [
    'DEFAULT_ALPHA',
    'DEFAULT_BOOTSTRAP_SAMPLES',
    'DEFAULT_CONFIDENCE',
    'DEFAULT_MIN_PAIRS',
    'AnalysisMetric',
    'Direction',
    'FriedmanTest',
    'PairedTest',
    'against_baseline',
    'annotate_tests',
    'average_ranks',
    'bootstrap_median',
    'correlations',
    'friedman',
    'holm_adjust',
    'load_results',
    'method_matrix',
    'missing_runs',
    'pairwise_wilcoxon',
    'per_unit_ranks',
    'rank_agreement',
    'rank_biserial',
    'stars',
    'wilcoxon_paired',
]

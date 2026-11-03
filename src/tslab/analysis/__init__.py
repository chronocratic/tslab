"""Post-hoc analysis of study results: tests, diagrams, numbers and an organised output tree.

Needs the ``analysis`` extra.
"""

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

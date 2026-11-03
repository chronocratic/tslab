"""Post-hoc analysis of study results: tests, diagrams, numbers and an organised output tree.

Needs the ``analysis`` extra.
"""

from tslab.analysis.cd_diagram import cd_diagram, CdDiagram, critical_difference
from tslab.analysis.output import AnalysisOutput, FigureFormat, OutputFolder, PLOTS_FOLDER
from tslab.analysis.report import Block, Caption, show
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
from tslab.analysis.style import FULL_WIDTH, HALF_WIDTH, is_dark_cell, paper_style

__all__ = [
    'DEFAULT_ALPHA',
    'DEFAULT_BOOTSTRAP_SAMPLES',
    'DEFAULT_CONFIDENCE',
    'DEFAULT_MIN_PAIRS',
    'FULL_WIDTH',
    'HALF_WIDTH',
    'PLOTS_FOLDER',
    'AnalysisMetric',
    'AnalysisOutput',
    'Block',
    'Caption',
    'CdDiagram',
    'Direction',
    'FigureFormat',
    'FriedmanTest',
    'OutputFolder',
    'PairedTest',
    'against_baseline',
    'annotate_tests',
    'average_ranks',
    'bootstrap_median',
    'cd_diagram',
    'correlations',
    'critical_difference',
    'friedman',
    'holm_adjust',
    'is_dark_cell',
    'load_results',
    'method_matrix',
    'missing_runs',
    'pairwise_wilcoxon',
    'paper_style',
    'per_unit_ranks',
    'rank_agreement',
    'rank_biserial',
    'show',
    'stars',
    'wilcoxon_paired',
]

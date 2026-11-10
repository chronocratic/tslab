"""Post-hoc analysis of study results: tests, diagrams, numbers and an organised output tree.

Needs the ``analysis`` extra.
"""

from tslab.lab.analysis.cd_diagram import cd_diagram, CdDiagram, CdStyle, critical_difference
from tslab.lab.analysis.numbers import (
    format_number,
    number_key,
    NumberKind,
    numbers_frame,
    NUMBERS_STEM,
    write_numbers,
)
from tslab.lab.analysis.output import AnalysisOutput, FigureFormat, OutputFolder, PLOTS_FOLDER
from tslab.lab.analysis.report import Block, Caption, show
from tslab.lab.analysis.results import (
    against_baseline,
    AnalysisMetric,
    load_results,
    method_matrix,
    missing_runs,
)
from tslab.lab.analysis.stats import (
    annotate_tests,
    average_ranks,
    bootstrap_median,
    complete_row_count,
    correlations,
    DEFAULT_ALPHA,
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_CONFIDENCE,
    DEFAULT_MIN_PAIRS,
    Direction,
    friedman,
    FriedmanTest,
    holm_adjust,
    PairedTest,
    pairwise_wilcoxon,
    per_unit_ranks,
    rank_agreement,
    rank_biserial,
    stars,
    wilcoxon_paired,
)
from tslab.lab.analysis.style import FULL_WIDTH, is_dark_cell, paper_style

__all__ = [
    'DEFAULT_ALPHA',
    'DEFAULT_BOOTSTRAP_SAMPLES',
    'DEFAULT_CONFIDENCE',
    'DEFAULT_MIN_PAIRS',
    'FULL_WIDTH',
    'NUMBERS_STEM',
    'PLOTS_FOLDER',
    'AnalysisMetric',
    'AnalysisOutput',
    'Block',
    'Caption',
    'CdDiagram',
    'CdStyle',
    'Direction',
    'FigureFormat',
    'FriedmanTest',
    'NumberKind',
    'OutputFolder',
    'PairedTest',
    'against_baseline',
    'annotate_tests',
    'average_ranks',
    'bootstrap_median',
    'cd_diagram',
    'complete_row_count',
    'correlations',
    'critical_difference',
    'format_number',
    'friedman',
    'holm_adjust',
    'is_dark_cell',
    'load_results',
    'method_matrix',
    'missing_runs',
    'number_key',
    'numbers_frame',
    'pairwise_wilcoxon',
    'paper_style',
    'per_unit_ranks',
    'rank_agreement',
    'rank_biserial',
    'show',
    'stars',
    'wilcoxon_paired',
    'write_numbers',
]

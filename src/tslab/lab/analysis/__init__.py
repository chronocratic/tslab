"""Post-hoc analysis of study results: tests, diagrams, numbers and an organized output tree.

Needs the ``analysis`` extra.
"""

from tslab.lab.analysis.cd_diagram import cd_diagram, CdDiagram, CdStyle, critical_difference
from tslab.lab.analysis.numbers import (
    number_key,
    NumberKind,
    numbers_frame,
    NUMBERS_STEM,
    write_numbers,
)
from tslab.lab.analysis.output import AnalysisOutput, FigureFormat, OutputFolder
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
    correlations,
    Direction,
    friedman,
    FriedmanTest,
    holm_adjust,
    PairedTest,
    pairwise_wilcoxon,
    per_unit_ranks,
    rank_agreement,
    rank_biserial,
    wilcoxon_paired,
)
from tslab.lab.analysis.style import paper_style

__all__ = [
    'NUMBERS_STEM',
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
    'correlations',
    'critical_difference',
    'friedman',
    'holm_adjust',
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
    'wilcoxon_paired',
    'write_numbers',
]

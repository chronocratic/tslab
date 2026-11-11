import tslab.lab.analysis

EXPECTED = {
    'AnalysisMetric',
    'AnalysisOutput',
    'Block',
    'Caption',
    'CdDiagram',
    'CdStyle',
    'Direction',
    'FigureFormat',
    'FriedmanTest',
    'NUMBERS_STEM',
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
}


def test_door_lists_the_public_api() -> None:
    assert set(tslab.lab.analysis.__all__) == EXPECTED
    assert all(hasattr(tslab.lab.analysis, name) for name in tslab.lab.analysis.__all__)

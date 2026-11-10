from dataclasses import dataclass
import os
from pathlib import Path
import sys

import pandas as pd
import pytest

from tests.unit.study.project import write_toy_project
from tslab.analysis.results import (
    against_baseline,
    AnalysisMetric,
    load_results,
    method_matrix,
    missing_runs,
)
from tslab.analysis.stats import Direction
from tslab.study import DEFAULT_RUN_SETTINGS, Grid, Study
from tslab.study.definition import load_study


@dataclass(frozen=True, kw_only=True)
class TinyConfig:
    """The config of a run with a single setting."""

    seed: int = 0
    output_root: Path = Path()
    run_path: Path = Path('run')


def some_study() -> Study:
    return Study(
        name='s',
        grids={
            'g': Grid(
                short_name='g1',
                command=('script.py',),
                config=TinyConfig,
                axes={'seed': {'1': {'seed': 1}, '2': {'seed': 2}}},
                run_settings=DEFAULT_RUN_SETTINGS,
            )
        },
    )


def test_metric_column() -> None:
    metric = AnalysisMetric(name='accuracy', label='Accuracy', direction=Direction.HIGHER_IS_BETTER)
    assert metric.column == 'metric/accuracy'


def test_load_results_runs_from_project_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Path] = {}

    def fake(*, study: Study, machine: Path, environment: object) -> list[dict[str, object]]:
        seen['cwd'], seen['machine'] = Path.cwd(), machine
        return [{'grid': 'g', 'instance': None}]

    monkeypatch.setattr('tslab.study.reporting.result_table.study_results', fake)
    (tmp_path / 'proj').mkdir()
    monkeypatch.chdir(tmp_path)
    frame = load_results(
        study=some_study(),
        machine=Path('local.toml'),
        project_root=tmp_path / 'proj',
        environment={},
    )
    assert seen == {'cwd': (tmp_path / 'proj').resolve(), 'machine': Path('local.toml')}
    assert Path.cwd() == tmp_path.resolve()
    assert len(frame) == 1


def test_load_results_restores_cwd_on_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake(**_kwargs: object) -> list[dict[str, object]]:
        message = 'boom'
        raise RuntimeError(message)

    monkeypatch.setattr('tslab.study.reporting.result_table.study_results', fake)
    (tmp_path / 'proj').mkdir()
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match='boom'):
        load_results(
            study=some_study(),
            machine=Path('local.toml'),
            project_root=tmp_path / 'proj',
            environment={},
        )
    assert Path.cwd() == tmp_path.resolve()


def test_load_results_rejects_str_paths() -> None:
    with pytest.raises(TypeError, match='machine'):
        load_results(
            study=some_study(),
            machine='local.toml',  # ty: ignore[invalid-argument-type]
            project_root=Path(),
            environment={},
        )


def test_load_results_reads_from_project_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = write_toy_project(project=tmp_path / 'proj', output_root=Path('runs'))
    monkeypatch.chdir(folder)
    for name in ('toy_studies', 'toy_script'):
        monkeypatch.delitem(sys.modules, name, raising=False)
    study = load_study(reference='toy_studies:study')
    monkeypatch.chdir(tmp_path)
    frame = load_results(
        study=study, machine=Path('local.toml'), project_root=folder, environment=dict(os.environ)
    )
    assert len(frame) == 4
    assert set(frame['study_state']) == {'missing'}
    assert len(missing_runs(results=frame)) == 4


def test_missing_runs_without_metrics() -> None:
    frame = pd.DataFrame(
        {
            'grid': ['g', 'g'],
            'run_path': ['g/a', 'g/b'],
            'instance': ['evaluate[default]', None],
            'instance_state': ['done', None],
            'study_state': ['done', 'missing'],
            'seconds': [1.0, None],
            'error': [None, None],
            'metric/accuracy': [0.9, None],
        }
    )
    result = missing_runs(results=frame)
    assert result['run_path'].tolist() == ['g/b']
    assert 'metric/accuracy' not in result.columns
    assert 'seconds' not in result.columns


def test_against_baseline_sign_for_both_directions() -> None:
    frame = pd.DataFrame(
        {'dataset': ['d', 'd'], 'noise': ['clean', 'noisy'], 'metric/m': [0.9, 0.8]}
    )
    for direction, expected in (
        (Direction.HIGHER_IS_BETTER, -0.1),
        (Direction.LOWER_IS_BETTER, 0.1),
    ):
        metric = AnalysisMetric(name='m', label='m', direction=direction)
        result = against_baseline(
            frame=frame, condition='noise', baseline='clean', keys=['dataset'], metric=metric
        )
        assert result['noise'].tolist() == ['noisy']
        assert result['baseline_value'].tolist() == [0.9]
        assert result['delta'].tolist() == pytest.approx([expected])


def test_against_baseline_duplicate_baseline_raises() -> None:
    frame = pd.DataFrame(
        {
            'dataset': ['d', 'd', 'd'],
            'noise': ['clean', 'clean', 'noisy'],
            'metric/m': [0.9, 0.7, 0.8],
        }
    )
    metric = AnalysisMetric(name='m', label='m', direction=Direction.HIGHER_IS_BETTER)
    with pytest.raises(ValueError, match='keys'):
        against_baseline(
            frame=frame, condition='noise', baseline='clean', keys=['dataset'], metric=metric
        )


def test_method_matrix_averages_and_raises_on_duplicates() -> None:
    frame = pd.DataFrame(
        {
            'dataset': ['d1'] * 4,
            'recipe': ['a', 'a', 'b', 'b'],
            'seed': [0, 1, 0, 1],
            'metric/m': [0.8, 0.6, 0.5, 0.5],
        }
    )
    matrix = method_matrix(
        frame=frame, unit='dataset', method='recipe', value='metric/m', average_over=['seed']
    )
    assert matrix.loc['d1'].to_dict() == pytest.approx({'a': 0.7, 'b': 0.5})
    doubled = pd.concat([frame, frame])
    with pytest.raises(ValueError, match='instance'):
        method_matrix(
            frame=doubled, unit='dataset', method='recipe', value='metric/m', average_over=['seed']
        )


def test_missing_runs_on_an_empty_frame() -> None:
    assert missing_runs(results=pd.DataFrame()).empty


def test_method_matrix_keeps_a_method_with_only_missing_scores() -> None:
    frame = pd.DataFrame(
        {
            'dataset': ['d1', 'd1', 'd2', 'd2'],
            'recipe': ['a', 'b', 'a', 'b'],
            'metric/m': [0.8, float('nan'), 0.7, float('nan')],
        }
    )
    matrix = method_matrix(
        frame=frame, unit='dataset', method='recipe', value='metric/m', average_over=[]
    )
    assert list(matrix.columns) == ['a', 'b']
    assert matrix['b'].isna().all()

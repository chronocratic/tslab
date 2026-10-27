"""The results table: one row per instance, every metric of every run."""

from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.unit.study.helpers import no_sleep
import tslab
from tslab.core.status import (
    InstanceRecord,
    InstanceState,
    RunState,
    RunStatusSnapshot,
    utc_now_iso,
    write_status,
)
from tslab.study import DEFAULT_RUN_SETTINGS, Grid, Study
from tslab.study.commands import submit
from tslab.study.context import load_context, Selection
from tslab.study.definition import expand_runs
from tslab.study.loading import load_study
from tslab.study.result_table import collect_results, write_results_csv
from tslab.study.state import collect_states, RunReport


@dataclass(frozen=True, kw_only=True)
class TinyConfig:
    """The config of a run with a single setting."""

    seed: int = 0
    output_root: Path = Path()
    run_path: Path = Path('run')


@tslab.stage(name='train', output=tslab.Json)
def train(*, run: tslab.Run, name: tslab.InstanceName | None = None) -> tslab.Json:
    return tslab.Json(data={'value': 1})


@tslab.stage(name='evaluate', output=tslab.Metrics)
def evaluate(*, run: tslab.Run, name: tslab.InstanceName | None = None) -> tslab.Metrics:
    return tslab.Metrics(data={'accuracy': 0.9, 'f1': 0.8})


def study() -> Study:
    return Study(
        name='s',
        grids={
            'g': Grid(
                short_name='g1',
                command=('script.py',),
                config=TinyConfig,
                axes={
                    'model': {'small': {'seed': 0}, 'large': {'seed': 1}},
                    'seed': {'1': {'seed': 1}, '2': {'seed': 2}},
                },
                run_settings=DEFAULT_RUN_SETTINGS,
            )
        },
    )


def make_run(*, root: Path, run_path: str, with_metrics: bool) -> None:
    """One real run folder, with a train stage and, optionally, an evaluate one."""
    with tslab.Run(
        config=TinyConfig(seed=int(run_path.rsplit('/', 1)[-1])),
        root=root,
        relative_path=Path(run_path),
    ) as run:
        train(run=run)
        if with_metrics:
            evaluate(run=run)


def reports_of(*, study_directory: Path) -> list[RunReport]:
    study_definition = study()
    return collect_states(
        study_directory=study_directory,
        runs=expand_runs(study=study_definition),
        jobs=[],
        job_states=None,
    )


def test_rows_per_instance_with_metrics(tmp_path: Path) -> None:
    make_run(root=tmp_path, run_path='g/small/1', with_metrics=True)
    rows = collect_results(
        study=study(), study_directory=tmp_path, reports=reports_of(study_directory=tmp_path)
    )
    assert {row['instance'] for row in rows} == {'train[default]', 'evaluate[default]', None}
    evaluate_row = next(row for row in rows if row['instance'] == 'evaluate[default]')
    assert evaluate_row['metric/accuracy'] == 0.9
    assert evaluate_row['metric/f1'] == 0.8
    assert evaluate_row['grid'] == 'g'
    assert evaluate_row['model'] == 'small'
    assert evaluate_row['seed'] == '1'
    assert evaluate_row['run_path'] == 'g/small/1'
    assert evaluate_row['study_state'] == 'done'
    assert evaluate_row['instance_state'] == 'done'
    assert evaluate_row['error'] is None
    train_row = next(row for row in rows if row['instance'] == 'train[default]')
    assert train_row['metric/accuracy'] is None


def test_runs_without_instances_get_one_row(tmp_path: Path) -> None:
    make_run(root=tmp_path, run_path='g/small/1', with_metrics=False)
    rows = collect_results(
        study=study(), study_directory=tmp_path, reports=reports_of(study_directory=tmp_path)
    )
    instances = [row['instance'] for row in rows]
    assert 'train[default]' in instances
    missing = [row for row in rows if row['instance'] is None]
    assert len(missing) == 3
    assert {row['study_state'] for row in missing} == {'missing'}
    assert {row['run_path'] for row in missing} == {'g/large/1', 'g/large/2', 'g/small/2'}
    assert all(row['error'] is None for row in missing)


def test_unreadable_files_become_unreadable_rows(tmp_path: Path) -> None:
    make_run(root=tmp_path, run_path='g/small/1', with_metrics=True)
    (tmp_path / 'g/small/1/evaluate' / 'default' / 'meta.json').write_text('{')
    rows = collect_results(
        study=study(), study_directory=tmp_path, reports=reports_of(study_directory=tmp_path)
    )
    evaluate_row = next(row for row in rows if row['instance'] == 'evaluate[default]')
    assert evaluate_row['instance_state'] == 'unreadable'
    assert 'JSON' in str(evaluate_row['error'])
    train_row = next(row for row in rows if row['instance'] == 'train[default]')
    assert train_row['instance_state'] == 'done'
    assert train_row['error'] is None


def test_columns_and_csv(tmp_path: Path) -> None:
    rows = [
        {
            'grid': 'g',
            'seed': '1',
            'run_path': 'g/1',
            'study_state': 'done',
            'instance': 'e[default]',
            'instance_state': 'done',
            'error': None,
            'metric/b': 2.0,
            'metric/a': 1.0,
        }
    ]
    write_results_csv(rows=rows, path=tmp_path / 'r.csv')
    header, line = (tmp_path / 'r.csv').read_text().splitlines()
    expected_header = (
        'grid,seed,run_path,study_state,instance,instance_state,error,metric/a,metric/b'
    )
    assert header == expected_header
    assert line == 'g,1,g/1,done,e[default],done,,1.0,2.0'


@pytest.mark.slow
def test_study_results_method(project: Path) -> None:
    code = submit(
        context=load_context(
            reference='toy_studies:study',
            machine_path=project / 'local.toml',
            selection=Selection(),
            project_root=project,
        ),
        include_done=False,
        dry_run=False,
        extra_args=(),
        sleep=no_sleep,
    )
    assert code == 0
    rows = load_study(reference='toy_studies:study').results(machine=Path('local.toml'))
    assert len(rows) == 4
    assert all(row['instance'] is None for row in rows)
    assert {row['study_state'] for row in rows} == {'done'}
    assert {row['run_path'] for row in rows} == {'g/small/1', 'g/small/2', 'g/large/1', 'g/large/2'}


def test_csv_repeats_no_metric_column_for_many_rows(tmp_path: Path) -> None:
    make_run(root=tmp_path, run_path='g/small/1', with_metrics=True)
    make_run(root=tmp_path, run_path='g/small/2', with_metrics=True)
    rows = collect_results(
        study=study(), study_directory=tmp_path, reports=reports_of(study_directory=tmp_path)
    )
    assert len(rows) > 2
    write_results_csv(rows=rows, path=tmp_path / 'r.csv')
    header = (tmp_path / 'r.csv').read_text().splitlines()[0].split(',')
    assert len(header) == len(set(header))
    assert header[-2:] == ['metric/accuracy', 'metric/f1']


def test_failed_and_running_instances_show_their_own_state(tmp_path: Path) -> None:
    folder = tmp_path / 'g/small/1'
    folder.mkdir(parents=True)
    write_status(
        path=folder / 'status.json',
        snapshot=RunStatusSnapshot(
            run_state=RunState.FAILED,
            updated_at=utc_now_iso(),
            instances={
                'train[default]': InstanceRecord(
                    state=InstanceState.FAILED, error='ValueError: bad\nTraceback'
                ),
                'evaluate[default]': InstanceRecord(state=InstanceState.RUNNING),
            },
        ),
    )
    rows = collect_results(
        study=study(), study_directory=tmp_path, reports=reports_of(study_directory=tmp_path)
    )
    by_instance = {row['instance']: row for row in rows if row['instance'] is not None}
    assert by_instance['train[default]']['instance_state'] == 'failed'
    assert by_instance['train[default]']['error'] == 'ValueError: bad'
    assert by_instance['evaluate[default]']['instance_state'] == 'running'
    assert by_instance['evaluate[default]']['error'] is None


def test_unreadable_status_file_is_an_unreadable_row(tmp_path: Path) -> None:
    folder = tmp_path / 'g/small/1'
    folder.mkdir(parents=True)
    (folder / 'status.json').write_text('{')
    rows = collect_results(
        study=study(), study_directory=tmp_path, reports=reports_of(study_directory=tmp_path)
    )
    row = next(row for row in rows if row['run_path'] == 'g/small/1')
    assert row['instance_state'] == 'unreadable'
    assert row['error'] is not None


def test_study_results_without_runs_lists_missing_rows(project: Path) -> None:
    rows = load_study(reference='toy_studies:study').results(machine=Path('local.toml'))
    assert len(rows) == 4
    assert {row['study_state'] for row in rows} == {'missing'}

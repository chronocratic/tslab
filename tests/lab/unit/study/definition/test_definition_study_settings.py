from dataclasses import dataclass
import logging
from pathlib import Path
from types import MappingProxyType

import pytest

from tests.lab.unit.study.configs import grid
from tslab.lab.study import Grid, Override, Study
from tslab.lab.study.definition.definition import expand_runs
from tslab.lab.study.reporting.documents import show_document
from tslab.lab.study.reporting.report import log_show


@dataclass(frozen=True, kw_only=True)
class ForecastConfig:
    output_root: Path = Path('runs')
    run_path: Path = Path('forecasting')
    seed: int = 0


def _grid_without_trainer_axis() -> Grid:
    return grid(axes={'seed': {'0': {'seed': 0}, '1': {'seed': 1}}})


def _forecast_grid() -> Grid:
    return grid(short_name='fc', config=ForecastConfig, axes={'seed': {'0': {'seed': 0}}})


def test_study_settings_reach_every_run() -> None:
    study = Study(
        name='s', settings={'trainer.max_epochs': 5}, grids={'g': _grid_without_trainer_axis()}
    )
    assert all(run.settings['trainer.max_epochs'] == 5 for run in expand_runs(study=study))


def test_study_setting_only_applies_to_grids_that_have_it() -> None:
    study = Study(
        name='s',
        settings={'noise_levels': (0.1,)},
        grids={'cls': _grid_without_trainer_axis(), 'fc': _forecast_grid()},
    )
    runs = expand_runs(study=study)
    assert all('noise_levels' in run.settings for run in runs if run.grid == 'cls')
    assert all('noise_levels' not in run.settings for run in runs if run.grid == 'fc')


def test_study_setting_on_no_grid_fails() -> None:
    with pytest.raises(
        ValueError, match=r"study 's' settings: trainer\.max_epoch is no field of any grid"
    ):
        Study(name='s', settings={'trainer.max_epoch': 5}, grids={'g': grid()})


def test_study_settings_must_be_a_mapping() -> None:
    with pytest.raises(TypeError, match='study settings must be a mapping'):
        Study(name='s', settings=(), grids={'g': grid()})  # ty: ignore[invalid-argument-type]


def test_study_settings_are_read_only_and_copied() -> None:
    settings = {'trainer.max_epochs': 5}
    study = Study(name='s', settings=settings, grids={'g': _grid_without_trainer_axis()})
    settings['trainer.max_epochs'] = 10
    assert study.settings['trainer.max_epochs'] == 5
    assert isinstance(study.settings, MappingProxyType)


def test_duplicate_with_preset_axis_fails() -> None:
    with pytest.raises(
        ValueError,
        match=(
            r"study 's' sets trainer\.max_epochs, also set by grid 'g', axis 'recipe', "
            r"value '.*'; keep it in one place, e\.g\. drop it from the axis with "
            r"without=\('trainer\.max_epochs',\)"
        ),
    ):
        Study(name='s', settings={'trainer.max_epochs': 5}, grids={'g': grid()})


@pytest.mark.parametrize(
    ('study_settings', 'axis_settings'),
    [
        ({'trainer': 'gpu'}, {'trainer.max_epochs': 5}),
        ({'trainer.max_epochs': 5}, {'trainer': 'gpu'}),
    ],
)
def test_duplicate_prefixes_fail_both_ways(
    study_settings: dict[str, int | str], axis_settings: dict[str, int | str]
) -> None:
    with pytest.raises(ValueError, match='also set by grid'):
        Study(
            name='s',
            settings=study_settings,
            grids={'g': grid(axes={'settings': {'one': axis_settings}})},
        )


def test_duplicate_with_run_settings_fails() -> None:
    with pytest.raises(ValueError, match=r"also set by grid 'g', run_settings"):
        Study(name='s', settings={'seed': 1}, grids={'g': grid(run_settings={'seed': 2})})


def test_study_setting_cannot_replace_default_run_settings() -> None:
    with pytest.raises(ValueError, match=r"also set by grid 'g', run_settings"):
        Study(
            name='s',
            settings={'output_root': 'elsewhere'},
            grids={'g': grid(run_settings={'trainer.max_epochs': 5})},
        )


def test_every_duplicate_is_listed() -> None:
    axis_settings = {'trainer.max_epochs': 5, 'trainer.max_training_time': '1:00:00'}
    with pytest.raises(ValueError, match=r'trainer\.max_epochs') as raised:
        Study(
            name='s',
            settings=axis_settings,
            grids={'g': grid(axes={'settings': {'one': axis_settings}})},
        )
    assert 'trainer.max_epochs' in str(raised.value)
    assert 'trainer.max_training_time' in str(raised.value)


def test_override_may_set_a_study_path() -> None:
    study = Study(
        name='s',
        settings={'trainer.max_epochs': 5},
        grids={
            'g': grid(
                axes={'seed': {'0': {'seed': 0}, '1': {'seed': 1}}},
                overrides=(Override(match={'seed': '0'}, settings={'trainer.max_epochs': 1}),),
            )
        },
    )
    by_seed = {
        run.values['seed']: run.settings['trainer.max_epochs'] for run in expand_runs(study=study)
    }
    assert by_seed == {'0': 1, '1': 5}


def test_run_settings_add_to_defaults() -> None:
    run = expand_runs(
        study=Study(name='s', grids={'g': grid(run_settings={'trainer.max_epochs': 5})})
    )[0]
    assert run.settings['output_root'] == '{output_root}/{study}'
    assert run.settings['run_path'] == '{run_path}'
    assert run.settings['trainer.max_epochs'] == 5


def test_show_document_lists_json_safe_study_settings() -> None:
    study = Study(
        name='s',
        settings={'trainer.max_epochs': 5, 'data.dataset_folder_path': Path('datasets')},
        grids={'g': _grid_without_trainer_axis()},
    )
    assert show_document(study=study)['settings'] == {
        'trainer.max_epochs': 5,
        'data.dataset_folder_path': 'datasets',
    }


def test_log_show_lists_study_settings_before_grids(caplog: pytest.LogCaptureFixture) -> None:
    study = Study(
        name='s', settings={'trainer.max_epochs': 5}, grids={'g': _grid_without_trainer_axis()}
    )
    with caplog.at_level(logging.INFO, logger='tslab.lab.study.reporting.report'):
        log_show(study=study)
    messages = [record.message for record in caplog.records]
    assert messages.index('study settings:') < next(
        position
        for position, message in enumerate(messages)
        if 'grid' in message and 'runs' in message
    )
    assert messages.count('trainer.max_epochs = 5') == 1

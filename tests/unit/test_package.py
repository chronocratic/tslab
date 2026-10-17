from importlib import import_module
from importlib.metadata import version
import re

import pytest

import tslab

OLD_MODULES = (
    'tslab.artifact',
    'tslab.config',
    'tslab.fingerprint',
    'tslab.meta',
    'tslab.paths',
    'tslab.run',
    'tslab.seeding',
    'tslab.stage',
    'tslab.status',
    'tslab.ts',
    'tslab.adapters.chronocratic_models',
    'tslab.adapters.chronocratic_datasets',
)

TS_NAMES = {
    'ForecastingSeries',
    'TSClassificationDataInfo',
    'TSDataInfo',
    'TSDataSource',
    'TSForecastingDataInfo',
    'checked_split_slices',
    'encode_series',
    'encode_windows',
    'forecasting_representations',
    'horizon_samples',
    'horizon_targets',
    'window_samples',
}


def test_version_comes_from_the_installed_package() -> None:
    assert tslab.__version__ == version('tslab')
    assert re.fullmatch(r'\d+\.\d+\.\d+.*', tslab.__version__)


@pytest.mark.parametrize('name', OLD_MODULES)
def test_old_module_paths_are_gone(name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        import_module(name)


def test_tasks_keeps_the_ts_names() -> None:
    tasks = import_module('tslab.tasks')
    assert set(tasks.__all__) == TS_NAMES
    assert tasks.TSForecastingDataInfo.__module__ == 'tslab.tasks.forecasting'
    assert tasks.TSClassificationDataInfo.__module__ == 'tslab.tasks.classification'
    assert tasks.TSDataInfo.__module__ == 'tslab.tasks.data'

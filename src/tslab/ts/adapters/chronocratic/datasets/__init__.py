"""Adapter for chronocratic-datasets (``pip install tslab[chronocratic-datasets]``).

Every chronocratic datamodule is offered on the command line with settings
generated from its constructor. ``build_chronocratic_data_source`` wraps it as a TSLab
data source; Lightning still receives the datamodule itself for training.
"""

from tslab.adapters.chronocratic.datasets.data import (
    build_chronocratic_data_source,
    ChronocraticClassificationData,
    ChronocraticForecastingData,
)
from tslab.adapters.chronocratic.datasets.settings import (
    ChronocraticDataChoice,
    data_settings,
    DATAMODULES,
)
from tslab.adapters.chronocratic.datasets.study import ucr_datasets, uea_datasets

__all__ = [
    'DATAMODULES',
    'ChronocraticClassificationData',
    'ChronocraticDataChoice',
    'ChronocraticForecastingData',
    'build_chronocratic_data_source',
    'data_settings',
    'ucr_datasets',
    'uea_datasets',
]

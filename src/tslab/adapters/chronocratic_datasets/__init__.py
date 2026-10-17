"""Adapter for chronocratic-datasets (``pip install tslab[chronocratic-datasets]``).

Every chronocratic datamodule is offered on the command line with settings
generated from its constructor. ``build_chronocratic_data_source`` wraps it as a TSLab
data source; Lightning still receives the datamodule itself for training.
"""

from tslab.adapters.chronocratic_datasets.data import (
    build_chronocratic_data_source,
    check_label_range,
    ChronocraticClassificationData,
    ChronocraticForecastingData,
)
from tslab.adapters.chronocratic_datasets.settings import (
    ChronocraticDataChoice,
    data_settings,
    datamodule_class_of,
    DataModuleEntry,
    DATAMODULES,
    EXCLUDED_PARAMETERS,
    NOT_FINGERPRINTED_PARAMETERS,
)

__all__ = [
    'DATAMODULES',
    'EXCLUDED_PARAMETERS',
    'NOT_FINGERPRINTED_PARAMETERS',
    'ChronocraticClassificationData',
    'ChronocraticDataChoice',
    'ChronocraticForecastingData',
    'DataModuleEntry',
    'build_chronocratic_data_source',
    'check_label_range',
    'data_settings',
    'datamodule_class_of',
]

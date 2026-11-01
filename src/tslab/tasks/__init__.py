"""The tasks package: dataset facts and the forecasting protocol.

The core never imports this package.
"""

from tslab.tasks.classification import check_label_range, TSClassificationDataInfo
from tslab.tasks.data import TSDataInfo, TSDataSource
from tslab.tasks.forecasting import (
    checked_split_slices,
    encode_series,
    encode_windows,
    forecasting_representations,
    ForecastingSeries,
    horizon_samples,
    horizon_targets,
    TSForecastingDataInfo,
    window_samples,
)

__all__ = [
    'ForecastingSeries',
    'TSClassificationDataInfo',
    'TSDataInfo',
    'TSDataSource',
    'TSForecastingDataInfo',
    'check_label_range',
    'checked_split_slices',
    'encode_series',
    'encode_windows',
    'forecasting_representations',
    'horizon_samples',
    'horizon_targets',
    'window_samples',
]

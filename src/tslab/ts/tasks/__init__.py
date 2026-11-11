"""The tasks package: dataset facts and the forecasting protocol.

The core never imports this package.
"""

from tslab.ts.tasks.classification import check_label_range, TSClassificationDataInfo
from tslab.ts.tasks.data import TSDataInfo
from tslab.ts.tasks.forecasting import (
    checked_split_slices,
    forecasting_representations,
    ForecastingSeries,
    horizon_targets,
    TSForecastingDataInfo,
    window_samples,
)

__all__ = [
    'ForecastingSeries',
    'TSClassificationDataInfo',
    'TSDataInfo',
    'TSForecastingDataInfo',
    'check_label_range',
    'checked_split_slices',
    'forecasting_representations',
    'horizon_targets',
    'window_samples',
]

"""The time-series layer: dataset facts and the forecasting protocol.

The core never imports this package.
"""

from tslab.ts.forecasting import (
    encode_series,
    encode_windows,
    forecasting_representations,
    horizon_samples,
    horizon_targets,
    window_samples,
)
from tslab.ts.info import (
    checked_split_slices,
    ForecastingSeries,
    TSClassificationDataInfo,
    TSDataInfo,
    TSDataSource,
    TSForecastingDataInfo,
)

__all__ = [
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
]

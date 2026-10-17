"""Facts about time series datasets, one class per task."""

from dataclasses import dataclass

from tslab.interfaces import DataSource

__all__ = ['TSDataInfo', 'TSDataSource']


@dataclass(frozen=True, kw_only=True)
class TSDataInfo:
    """Facts true for every time series dataset.

    Attributes:
        num_channels: Channels per timestep (variables, plus time features).
        length: Timesteps per series, or ``None`` for variable-length data.
    """

    num_channels: int
    length: int | None


# A data source whose facts are time series facts.
type TSDataSource[I: TSDataInfo] = DataSource[I]

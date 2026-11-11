"""Facts about time series datasets, one class per task."""

from dataclasses import dataclass

from tslab.ml.interfaces.data import DataSource

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

    def __post_init__(self) -> None:
        """Check the facts.

        Raises:
            ValueError: If ``num_channels`` is below 1 or ``length`` is below 1.
        """
        if self.num_channels < 1:
            message = f'num_channels must be at least 1, got {self.num_channels}'
            raise ValueError(message)
        if self.length is not None and self.length < 1:
            message = f'length must be at least 1 or None, got {self.length}'
            raise ValueError(message)


# A data source whose facts are time series facts.
type TSDataSource[I: TSDataInfo] = DataSource[I]

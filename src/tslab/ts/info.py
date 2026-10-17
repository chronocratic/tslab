"""Facts about time-series datasets, one class per task."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import torch

from tslab.interfaces import DataSource, Split, SPLITS

__all__ = [
    'ForecastingSeries',
    'TSClassificationDataInfo',
    'TSDataInfo',
    'TSDataSource',
    'TSForecastingDataInfo',
    'checked_split_slices',
]


@dataclass(frozen=True)
class TSDataInfo:
    """Facts true for every time-series dataset.

    Attributes:
        num_channels: Channels per timestep (variables, plus time features).
        length: Timesteps per series, or ``None`` for variable-length data.
    """

    num_channels: int
    length: int | None


@dataclass(frozen=True)
class TSClassificationDataInfo(TSDataInfo):
    """Facts about a classification dataset.

    Attributes:
        num_classes: Number of classes; labels are ``0 .. num_classes - 1``.
    """

    num_classes: int


@dataclass(frozen=True)
class TSForecastingDataInfo(TSDataInfo):
    """Facts about a forecasting dataset.

    Attributes:
        target_channels: Channels to forecast.
        covariate_channels: Channels that are known inputs but never targets,
            such as time features.
    """

    target_channels: tuple[int, ...]
    covariate_channels: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """Check the channel lists against each other and ``num_channels``.

        Raises:
            ValueError: If there is no target channel, an index is out of range
                or repeated, or a channel is both target and covariate.
        """
        if not self.target_channels:
            message = 'a forecasting dataset needs at least one target channel'
            raise ValueError(message)
        channels = (*self.target_channels, *self.covariate_channels)
        outside = [channel for channel in channels if not 0 <= channel < self.num_channels]
        if outside:
            message = f'channels {outside} are outside range({self.num_channels})'
            raise ValueError(message)
        if len(set(self.target_channels)) != len(self.target_channels) or len(
            set(self.covariate_channels)
        ) != len(self.covariate_channels):
            message = 'target_channels and covariate_channels must not contain duplicates'
            raise ValueError(message)
        both = sorted(set(self.target_channels) & set(self.covariate_channels))
        if both:
            message = f'channels {both} are both target and covariate'
            raise ValueError(message)


# A data source whose facts are time-series facts.
type TSDataSource[I: TSDataInfo] = DataSource[I]


class ForecastingSeries(Protocol):
    """A forecasting dataset as one long series plus the split boundaries.

    The TS2Vec forecasting protocol (Yue et al., AAAI 2022) encodes the whole
    series once, so later splits see earlier steps as context; it then cuts
    each split out of the result.
    """

    @property
    def info(self) -> TSForecastingDataInfo:
        """Return facts about the dataset."""
        ...

    def full_series(self) -> torch.Tensor:
        """Return every series in full, shape ``(N, T, C)``, covariates included."""
        ...

    def split_slices(self) -> Mapping[Split, slice]:
        """Return the timestep range of each split, keyed ``train``, ``val``, ``test``."""
        ...


def checked_split_slices(*, data: ForecastingSeries) -> dict[Split, slice]:
    """Return the split slices of ``data`` after checking them.

    Args:
        data: The forecasting data source.

    Returns:
        The slices, keyed ``train``, ``val`` and ``test``.

    Raises:
        ValueError: If a split is missing or extra, or a slice has a step
            other than 1.
    """
    slices = dict(data.split_slices())
    if sorted(slices) != sorted(SPLITS):
        message = f'split_slices() must have the keys {list(SPLITS)}, got {sorted(slices)}'
        raise ValueError(message)
    for split, window in slices.items():
        if window.step not in {None, 1}:
            message = f'the {split!r} slice has step {window.step}; only contiguous splits work'
            raise ValueError(message)
    return slices

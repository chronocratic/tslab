"""chronocratic datamodules as TSLab data sources."""

from collections.abc import Mapping
import dataclasses
from typing import Any

from chronocratic.datasets import (
    BaseClassificationTimeSeriesDataModule,
    BaseForecastingTimeSeriesDataModule,
    ClassificationLoaderMode,
)
import numpy as np
import torch

from tslab.adapters.chronocratic.datasets.settings import datamodule_class_of
from tslab.interfaces import Split
from tslab.lightning import LightningDataSource
from tslab.tasks import check_label_range, TSClassificationDataInfo, TSForecastingDataInfo

__all__ = [
    'ChronocraticClassificationData',
    'ChronocraticForecastingData',
    'build_chronocratic_data_source',
]


def _known[T](*, value: T | None, name: str) -> T:
    if value is None:
        message = f'the datamodule did not set {name} during setup'
        raise RuntimeError(message)
    return value


def _set_up(
    *, datamodule: BaseClassificationTimeSeriesDataModule | BaseForecastingTimeSeriesDataModule
) -> None:
    datamodule.prepare_data()
    datamodule.setup(stage='fit')


class ChronocraticClassificationData(LightningDataSource[TSClassificationDataInfo]):
    """A chronocratic classification datamodule (UCR, UEA) as a TSLab data source.

    Lightning receives the datamodule itself for training. ``labeled_split``
    returns each split exactly once, for encoding.
    """

    def __init__(
        self, *, datamodule: BaseClassificationTimeSeriesDataModule, settings: object
    ) -> None:
        """Load the data now, so ``info`` is known before anything trains.

        Args:
            datamodule: The chronocratic datamodule.
            settings: The settings it was built from; they are its fingerprint.
        """
        _set_up(datamodule=datamodule)
        info = TSClassificationDataInfo(
            num_channels=_known(value=datamodule.num_features, name='num_features'),
            length=datamodule.sequence_length,
            num_classes=_known(value=datamodule.num_classes, name='num_classes'),
        )
        super().__init__(datamodule=datamodule, info=info)
        self._record_set_up(stage='fit')
        self._classification = datamodule
        self._settings = settings

    def fingerprint_settings(self) -> object:
        """Return the settings the datamodule was built from."""
        return self._settings

    def labeled_split(self, *, split: Split) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the samples ``(N, T, C)`` and labels ``(N,)`` of ``split``, each sample once.

        An empty validation split gives empty tensors.

        Raises:
            ValueError: If a label lies outside ``0 .. num_classes - 1``.
        """
        self._set_up(stage='test' if split == 'test' else 'fit')
        loaders = {
            'train': self._classification.train_dataloader,
            'val': self._classification.val_dataloader,
            'test': self._classification.test_dataloader,
        }
        # Strict batches repeat tail samples to fill the last batch; collection must be exact.
        loader = loaders[split](
            loader_mode=ClassificationLoaderMode.SAMPLE_LABEL, loader_strict_batch_size=False
        )
        if loader is None and split == 'val':
            # A validation size of 0 leaves no validation loader: the split is empty.
            samples = torch.empty(0, self.info.length or 0, self.info.num_channels)
            return samples, torch.empty(0, dtype=torch.long)
        loader = _known(value=loader, name=f'{split}_dataloader')
        batches = list(loader)
        samples = torch.cat([batch_samples for batch_samples, _ in batches])
        labels = torch.cat([batch_labels for _, batch_labels in batches]).long()
        check_label_range(labels=labels, num_classes=self.info.num_classes, split=split)
        return samples, labels


class ChronocraticForecastingData(LightningDataSource[TSForecastingDataInfo]):
    """A chronocratic forecasting datamodule (ETT, Electricity, Weather) as a TSLab data source.

    chronocratic puts the time features first, so they are the covariates and
    the remaining channels are the targets.
    """

    def __init__(
        self, *, datamodule: BaseForecastingTimeSeriesDataModule, settings: object
    ) -> None:
        """Load the data now, so ``info`` is known before anything trains.

        Args:
            datamodule: The chronocratic datamodule.
            settings: The settings it was built from; they are its fingerprint.
        """
        _set_up(datamodule=datamodule)
        channels = int(np.asarray(datamodule.full_data).shape[-1])
        time_features = datamodule.num_time_series_features or 0
        info = TSForecastingDataInfo(
            num_channels=channels,
            length=None,
            target_channels=tuple(range(time_features, channels)),
            covariate_channels=tuple(range(time_features)),
        )
        super().__init__(datamodule=datamodule, info=info)
        self._record_set_up(stage='fit')
        self._forecasting = datamodule
        self._settings = settings

    def fingerprint_settings(self) -> object:
        """Return the settings the datamodule was built from."""
        return self._settings

    def full_series(self) -> torch.Tensor:
        """Return the whole series ``(1, T, C)``, time features first."""
        return torch.from_numpy(np.asarray(self._forecasting.full_data, dtype=np.float32))

    def split_slices(self) -> Mapping[Split, slice]:
        """Return the timestep range of each split."""
        return {
            'train': _known(value=self._forecasting.train_slice, name='train_slice'),
            'val': _known(value=self._forecasting.valid_slice, name='valid_slice'),
            'test': _known(value=self._forecasting.test_slice, name='test_slice'),
        }


def build_chronocratic_data_source(
    *,
    config: Any,  # noqa: ANN401 # why: generated settings classes have no static type
) -> ChronocraticClassificationData | ChronocraticForecastingData:
    """Build the chronocratic datamodule that ``config`` describes, set up and wrapped.

    Args:
        config: A dataset chosen through ``ChronocraticDataChoice``.

    Returns:
        The classification or forecasting data source.

    Raises:
        TypeError: If ``config`` is not chronocratic data settings, or the
            datamodule is neither a classification nor a forecasting one.
    """
    datamodule_class = datamodule_class_of(settings=config)
    datamodule = datamodule_class(
        **{field.name: getattr(config, field.name) for field in dataclasses.fields(config)}
    )
    if isinstance(datamodule, BaseClassificationTimeSeriesDataModule):
        return ChronocraticClassificationData(datamodule=datamodule, settings=config)
    if isinstance(datamodule, BaseForecastingTimeSeriesDataModule):
        return ChronocraticForecastingData(datamodule=datamodule, settings=config)
    message = (
        f'{datamodule_class.__qualname__} is neither a classification nor a forecasting datamodule'
    )
    raise TypeError(message)

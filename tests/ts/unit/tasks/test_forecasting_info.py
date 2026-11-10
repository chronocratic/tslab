from collections.abc import Mapping

import pytest
import torch

from tslab.ml.split import Split
from tslab.ts.tasks import checked_split_slices, TSForecastingDataInfo


def test_ts_forecasting_data_info_rejects_zero_channels() -> None:
    with pytest.raises(ValueError, match='num_channels'):
        TSForecastingDataInfo(num_channels=0, length=None, target_channels=(0,))


def test_forecasting_info_accepts_disjoint_channels() -> None:
    info = TSForecastingDataInfo(
        num_channels=4, length=100, target_channels=(2, 3), covariate_channels=(0, 1)
    )
    assert info.target_channels == (2, 3)


@pytest.mark.parametrize(
    ('targets', 'covariates', 'message'),
    [
        ((), (), 'at least one target'),
        ((4,), (), 'outside'),
        ((1, 1), (), 'duplicate'),
        ((1,), (1,), 'both'),
        ((-1,), (), 'outside'),
    ],
)
def test_forecasting_info_rejects_bad_channels(
    targets: tuple[int, ...], covariates: tuple[int, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        TSForecastingDataInfo(
            num_channels=4, length=100, target_channels=targets, covariate_channels=covariates
        )


class Series:
    def __init__(self, *, slices: Mapping[Split, slice]) -> None:
        self._slices = slices

    @property
    def info(self) -> TSForecastingDataInfo:
        return TSForecastingDataInfo(num_channels=1, length=10, target_channels=(0,))

    def full_series(self) -> torch.Tensor:
        return torch.zeros(1, 10, 1)

    def split_slices(self) -> Mapping[Split, slice]:
        return self._slices


def test_split_slices_are_checked() -> None:
    good: Mapping[Split, slice] = {
        Split.TRAIN: slice(0, 6),
        Split.VAL: slice(6, 8),
        Split.TEST: slice(8, 10),
    }
    assert checked_split_slices(data=Series(slices=good)) == good
    with pytest.raises(ValueError, match="'val'"):
        checked_split_slices(
            data=Series(slices={Split.TRAIN: slice(0, 6), Split.TEST: slice(6, 10)})
        )
    with pytest.raises(ValueError, match='step'):
        checked_split_slices(data=Series(slices={**good, Split.TEST: slice(8, 10, 2)}))

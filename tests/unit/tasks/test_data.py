import pytest

from tslab.tasks import TSClassificationDataInfo, TSDataInfo, TSForecastingDataInfo


def test_dataset_facts_are_validated() -> None:
    with pytest.raises(ValueError, match='num_channels'):
        TSDataInfo(num_channels=0, length=10)
    with pytest.raises(ValueError, match='length'):
        TSDataInfo(num_channels=1, length=0)
    with pytest.raises(ValueError, match='num_classes'):
        TSClassificationDataInfo(num_channels=1, length=None, num_classes=1)
    with pytest.raises(ValueError, match='num_channels'):
        TSForecastingDataInfo(num_channels=0, length=None, target_channels=(0,))

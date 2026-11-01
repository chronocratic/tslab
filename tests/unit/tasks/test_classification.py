import pytest

from tslab.tasks import TSClassificationDataInfo


def test_ts_classification_data_info_requires_num_classes() -> None:
    with pytest.raises(ValueError, match='num_classes'):
        TSClassificationDataInfo(num_channels=1, length=None, num_classes=1)

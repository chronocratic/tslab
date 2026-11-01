import pytest
import torch

from tslab.tasks import check_label_range, TSClassificationDataInfo


def test_ts_classification_data_info_requires_num_classes() -> None:
    with pytest.raises(ValueError, match='num_classes'):
        TSClassificationDataInfo(num_channels=1, length=None, num_classes=1)


def test_check_label_range() -> None:
    check_label_range(labels=torch.tensor([0, 1, 1]), num_classes=2, split='train')
    with pytest.raises(ValueError, match=r'outside \[0, 2\)'):
        check_label_range(labels=torch.tensor([1, 2]), num_classes=2, split='test')

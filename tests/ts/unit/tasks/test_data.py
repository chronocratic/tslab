import pytest

from tslab.ts.tasks import TSDataInfo


def test_ts_data_info_rejects_bad_facts() -> None:
    with pytest.raises(ValueError, match='num_channels'):
        TSDataInfo(num_channels=0, length=10)
    with pytest.raises(ValueError, match='length'):
        TSDataInfo(num_channels=1, length=0)

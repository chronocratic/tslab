import pytest

from tslab.enums import Split
from tslab.interfaces import SPLITS


def test_split_members_equal_their_text() -> None:
    assert Split.TRAIN == 'train'
    assert {'val': 1}[Split.VAL] == 1


def test_splits_lists_the_members_in_order() -> None:
    assert SPLITS == (Split.TRAIN, Split.VAL, Split.TEST)


def test_split_rejects_an_unknown_name() -> None:
    with pytest.raises(ValueError, match="'dev' is not a valid Split"):
        Split('dev')

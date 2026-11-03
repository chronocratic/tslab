"""The splits of a dataset."""

from enum import StrEnum

__all__ = ['Split']


class Split(StrEnum):
    """The three splits every ready stage works with; the text is the split name."""

    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'

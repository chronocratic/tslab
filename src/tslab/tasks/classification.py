"""The classification task: facts about a classification dataset."""

from dataclasses import dataclass

from tslab.tasks.data import TSDataInfo

__all__ = ['TSClassificationDataInfo']


@dataclass(frozen=True, kw_only=True)
class TSClassificationDataInfo(TSDataInfo):
    """Facts about a classification dataset.

    Attributes:
        num_classes: Number of classes; labels are ``0 .. num_classes - 1``.
    """

    num_classes: int

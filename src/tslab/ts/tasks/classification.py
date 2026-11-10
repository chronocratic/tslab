"""The classification task: facts about a classification dataset."""

from dataclasses import dataclass

import torch

from tslab.ml.split import Split
from tslab.ts.tasks.data import TSDataInfo

__all__ = ['TSClassificationDataInfo', 'check_label_range']

_MIN_CLASSES = 2


def check_label_range(*, labels: torch.Tensor, num_classes: int, split: Split) -> None:
    """Fail when a label lies outside ``0 .. num_classes - 1``.

    Raises:
        ValueError: If a label is negative or not below ``num_classes``.
    """
    if labels.numel() == 0:
        return
    lowest, highest = int(labels.min()), int(labels.max())
    if lowest < 0 or highest >= num_classes:
        message = (
            f'{split} labels lie in [{lowest}, {highest}], outside [0, {num_classes}); '
            'labels must be encoded as 0 .. num_classes - 1'
        )
        raise ValueError(message)


@dataclass(frozen=True, kw_only=True)
class TSClassificationDataInfo(TSDataInfo):
    """Facts about a classification dataset.

    Attributes:
        num_classes: Number of classes; labels are ``0 .. num_classes - 1``.
    """

    num_classes: int

    def __post_init__(self) -> None:
        """Check the facts.

        Raises:
            ValueError: If a base fact is invalid or ``num_classes`` is below 2.
        """
        super().__post_init__()
        if self.num_classes < _MIN_CLASSES:
            message = f'num_classes must be at least {_MIN_CLASSES}, got {self.num_classes}'
            raise ValueError(message)

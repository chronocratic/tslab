"""The classification task: facts about a classification dataset."""

from dataclasses import dataclass

from tslab.tasks.data import TSDataInfo

__all__ = ['TSClassificationDataInfo']

_MIN_CLASSES = 2


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

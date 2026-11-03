"""The data interface: what trainers and the ready stages need from a dataset."""

from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol

import torch

from tslab.core.enums import Split

__all__ = ['SPLITS', 'Batch', 'DataSource', 'LabeledSplits']

# One batch as a loader yields it: a tensor, a sequence of tensors or named tensors.
type Batch = Mapping[str, torch.Tensor] | Sequence[torch.Tensor] | torch.Tensor

# The splits in the order the ready stages work through them.
SPLITS: tuple[Split, ...] = tuple(Split)


class DataSource[I](Protocol):
    """A dataset split into train, validation and test loaders, plus facts about it.

    ``I`` is the type of ``info``: facts every consumer may rely on, such as the
    number of channels. Any object with these four members is a data source;
    nothing has to inherit from this class.
    """

    @property
    def info(self) -> I:
        """Return facts about the dataset."""
        ...

    def train_loader(self) -> Iterable[Batch]:
        """Return the training loader."""
        ...

    def val_loader(self) -> Iterable[Batch] | None:
        """Return the validation loader, or ``None`` if the dataset has no validation split."""
        ...

    def test_loader(self) -> Iterable[Batch]:
        """Return the test loader."""
        ...


class LabeledSplits(Protocol):
    """Data that hands out each split as two tensors, for encoding.

    Training loaders are the wrong source for encoding: they may shuffle, pad
    the last batch by repeating samples, or yield augmented pairs. A data
    source used by the ready ``encode`` stage therefore provides each split
    directly: every sample exactly once, in a fixed order.
    """

    def labeled_split(self, *, split: Split) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(inputs, targets)`` for ``split``.

        Args:
            split: A ``Split`` member.

        Returns:
            The inputs (one row per sample) and the matching targets.
        """
        ...

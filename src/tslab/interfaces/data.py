"""The data interface: what trainers and the ready stages need from a dataset."""

from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol

import torch

__all__ = ['Batch', 'DataSource']

# One batch as a loader yields it: a tensor, a sequence of tensors or named tensors.
type Batch = Mapping[str, torch.Tensor] | Sequence[torch.Tensor] | torch.Tensor


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

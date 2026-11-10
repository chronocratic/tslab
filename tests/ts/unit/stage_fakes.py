"""Fakes for the ready-stage tests: no Lightning, no chronocratic, all CPU."""

from collections.abc import Iterable, Mapping
from typing import cast

import torch

from tslab.ml.interfaces import Batch
from tslab.ml.split import Split
from tslab.ts.tasks import TSClassificationDataInfo, TSForecastingDataInfo


class LinearEncoder(torch.nn.Module):
    """Flattens each series and projects it to 4 values; ``dead`` returns NaN."""

    def __init__(self, *, dead: bool = False) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(in_features=10, out_features=4)
        self.dead = dead

    def fingerprint_settings(self) -> dict[str, bool]:
        return {'dead': self.dead}

    def encode(self, *, inputs: torch.Tensor) -> torch.Tensor:
        # why: torch ships no stub for Module.__call__, so the result is Any
        encoded = cast('torch.Tensor', self.layer(inputs.flatten(start_dim=1)))
        return torch.full_like(encoded, fill_value=torch.nan) if self.dead else encoded


class VectorEncoder(torch.nn.Module):
    """One vector per series (mean over time); counts its calls."""

    def __init__(self) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        self.weight = torch.nn.Parameter(torch.randn(3, 2, generator=generator))
        self.vector_calls = 0

    def fingerprint_settings(self) -> str:
        return type(self).__name__

    def encode(self, *, inputs: torch.Tensor) -> torch.Tensor:
        self.vector_calls += 1
        return (inputs @ self.weight).mean(dim=1)


class CausalEncoder(VectorEncoder):
    """Adds a causal per-timestep encoding: the running mean of a projection."""

    def __init__(self) -> None:
        super().__init__()
        self.sequence_calls = 0

    def encode_sequence(self, *, inputs: torch.Tensor) -> torch.Tensor:
        self.sequence_calls += 1
        steps = torch.arange(1, inputs.shape[1] + 1, dtype=inputs.dtype)[None, :, None]
        return (inputs @ self.weight).cumsum(dim=1) / steps


class ToySplits:
    """Two well-separated classes of series (5 steps, 2 channels).

    The training loaders raise: the ready encode stage must never use them.
    """

    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(0)
        self._splits: dict[Split, tuple[torch.Tensor, torch.Tensor]] = {}
        for split, count in ((Split.TRAIN, 40), (Split.VAL, 20), (Split.TEST, 20)):
            labels = torch.arange(count) % 2
            inputs = 4.0 * labels[:, None, None] + torch.randn(count, 5, 2, generator=generator)
            self._splits[split] = (inputs, labels)

    @property
    def info(self) -> TSClassificationDataInfo:
        return TSClassificationDataInfo(num_channels=2, length=5, num_classes=2)

    def fingerprint_settings(self) -> str:
        return 'toy-splits'

    def labeled_split(self, *, split: Split) -> tuple[torch.Tensor, torch.Tensor]:
        return self._splits[split]

    def train_loader(self) -> Iterable[Batch]:
        message = 'training loaders must not be used for encoding'
        raise AssertionError(message)

    def val_loader(self) -> Iterable[Batch] | None:
        message = 'training loaders must not be used for encoding'
        raise AssertionError(message)

    def test_loader(self) -> Iterable[Batch]:
        message = 'training loaders must not be used for encoding'
        raise AssertionError(message)


class ToySeries:
    """Two series of 40 steps: channel 0 is a time feature, channels 1 and 2 are targets."""

    def __init__(self) -> None:
        self._series = torch.randn(2, 40, 3, generator=torch.Generator().manual_seed(1))

    @property
    def info(self) -> TSForecastingDataInfo:
        return TSForecastingDataInfo(
            num_channels=3, length=40, target_channels=(1, 2), covariate_channels=(0,)
        )

    def fingerprint_settings(self) -> str:
        return 'toy-series'

    def full_series(self) -> torch.Tensor:
        return self._series

    def split_slices(self) -> Mapping[Split, slice]:
        return {Split.TRAIN: slice(0, 24), Split.VAL: slice(24, 32), Split.TEST: slice(32, 40)}

    # The fake trainer never reads batches; these make ToySeries a DataSource for train.
    def train_loader(self) -> Iterable[Batch]:
        batches: list[Batch] = []
        return batches

    def val_loader(self) -> Iterable[Batch] | None:
        return None

    def test_loader(self) -> Iterable[Batch]:
        batches: list[Batch] = []
        return batches

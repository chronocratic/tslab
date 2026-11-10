"""Tiny time series models and datasets for the end-to-end tests (CPU, seconds)."""

from collections.abc import Iterable, Mapping
import math
from typing import cast, override

from lightning import LightningModule
import torch
from torch.utils.data import DataLoader, TensorDataset

from tslab.ml.interfaces import Batch
from tslab.ml.split import Split
from tslab.ts.tasks import TSClassificationDataInfo, TSForecastingDataInfo


class ToyVectorModel(LightningModule):
    """Causal 1-D convolution; ``encode`` averages its output over time.

    With ``num_classes`` it trains a classifier head on that average; without,
    it trains to predict the next step of ``target_channel`` from each step.
    """

    def __init__(
        self, *, in_channels: int, num_classes: int | None = None, target_channel: int = 0
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.target_channel = target_channel
        self.convolution = torch.nn.Conv1d(in_channels=in_channels, out_channels=8, kernel_size=3)
        self.head = torch.nn.Linear(in_features=8, out_features=num_classes or 1)

    def fingerprint_settings(self) -> dict[str, int | str | None]:
        return {
            'model': type(self).__name__,
            'in_channels': self.in_channels,
            'num_classes': self.num_classes,
            'target_channel': self.target_channel,
        }

    def steps(self, *, inputs: torch.Tensor) -> torch.Tensor:
        # (N, T, C) -> (N, T, 8); left padding keeps every step causal.
        padded = torch.nn.functional.pad(input=inputs.transpose(1, 2), pad=(2, 0))
        return torch.relu(self.convolution(padded)).transpose(1, 2)

    def encode(self, *, inputs: torch.Tensor) -> torch.Tensor:
        return self.steps(inputs=inputs).mean(dim=1)

    def loss(self, *, batch: list[torch.Tensor]) -> torch.Tensor:
        if self.num_classes is not None:
            inputs, labels = batch
            logits = self.head(self.encode(inputs=inputs))
            return torch.nn.functional.cross_entropy(input=logits, target=labels)
        (inputs,) = batch
        predictions = self.head(self.steps(inputs=inputs)[:, :-1]).squeeze(-1)
        return torch.nn.functional.mse_loss(
            input=predictions, target=inputs[:, 1:, self.target_channel]
        )

    @override
    def training_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self.loss(batch=batch)
        self.log(name='train_loss', value=loss, on_step=True, on_epoch=True)
        return loss

    @override
    def validation_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self.loss(batch=batch)
        self.log(name='val_loss', value=loss, on_step=True, on_epoch=True)
        return loss

    @override
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(params=self.parameters(), lr=0.01)


class ToySequenceModel(ToyVectorModel):
    """``ToyVectorModel`` that also offers its per-step (causal) encoding."""

    def encode_sequence(self, *, inputs: torch.Tensor) -> torch.Tensor:
        return self.steps(inputs=inputs)


def _loader(*, tensors: tuple[torch.Tensor, ...], shuffle: bool) -> Iterable[Batch]:
    # torch types the batches of a TensorDataset as Any, so the loader is cast.
    return cast(
        'Iterable[Batch]',
        DataLoader(
            dataset=TensorDataset(*tensors),
            batch_size=16,
            shuffle=shuffle,
            generator=torch.Generator().manual_seed(0),
        ),
    )


class SineClassification:
    """Class 0: one sine period per 16 steps; class 1: per 4 steps; plus noise."""

    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(0)
        time = torch.arange(32, dtype=torch.float32)
        self._splits: dict[Split, tuple[torch.Tensor, torch.Tensor]] = {}
        for split, count in ((Split.TRAIN, 64), (Split.VAL, 32), (Split.TEST, 32)):
            labels = torch.arange(count) % 2
            periods = torch.where(labels == 0, 16.0, 4.0)[:, None]
            phases = 2 * math.pi * torch.rand(count, 1, generator=generator)
            clean = torch.sin(2 * math.pi * time[None, :] / periods + phases)
            noisy = clean + 0.1 * torch.randn(count, 32, generator=generator)
            self._splits[split] = (noisy[:, :, None], labels)

    @property
    def info(self) -> TSClassificationDataInfo:
        return TSClassificationDataInfo(num_channels=1, length=32, num_classes=2)

    def fingerprint_settings(self) -> str:
        return 'sine-classification'

    def labeled_split(self, *, split: Split) -> tuple[torch.Tensor, torch.Tensor]:
        return self._splits[split]

    def train_loader(self) -> Iterable[Batch]:
        return _loader(tensors=self._splits[Split.TRAIN], shuffle=True)

    def val_loader(self) -> Iterable[Batch] | None:
        return _loader(tensors=self._splits[Split.VAL], shuffle=False)

    def test_loader(self) -> Iterable[Batch]:
        return _loader(tensors=self._splits[Split.TEST], shuffle=False)


class SineForecasting:
    """One series of 400 steps: channel 0 is time (0..1), channel 1 a noisy sine."""

    SLICES: Mapping[Split, slice] = {
        Split.TRAIN: slice(0, 240),
        Split.VAL: slice(240, 320),
        Split.TEST: slice(320, 400),
    }

    def __init__(self) -> None:
        time = torch.arange(400, dtype=torch.float32)
        noise = 0.05 * torch.randn(400, generator=torch.Generator().manual_seed(0))
        target = torch.sin(2 * math.pi * time / 25) + noise
        self._series = torch.stack(tensors=[time / 400, target], dim=-1)[None]

    @property
    def info(self) -> TSForecastingDataInfo:
        return TSForecastingDataInfo(
            num_channels=2, length=400, target_channels=(1,), covariate_channels=(0,)
        )

    def fingerprint_settings(self) -> str:
        return 'sine-forecasting'

    def full_series(self) -> torch.Tensor:
        return self._series

    def split_slices(self) -> Mapping[Split, slice]:
        return self.SLICES

    def _windows(self, *, split: Split) -> tuple[torch.Tensor]:
        part = self._series[0, self.SLICES[split]]
        return (part.unfold(dimension=0, size=24, step=4).transpose(1, 2),)

    def train_loader(self) -> Iterable[Batch]:
        return _loader(tensors=self._windows(split=Split.TRAIN), shuffle=True)

    def val_loader(self) -> Iterable[Batch] | None:
        return _loader(tensors=self._windows(split=Split.VAL), shuffle=False)

    def test_loader(self) -> Iterable[Batch]:
        return _loader(tensors=self._windows(split=Split.TEST), shuffle=False)

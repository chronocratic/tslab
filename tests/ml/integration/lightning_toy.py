"""A tiny regression model and dataset for the Lightning tests (CPU, about a second)."""

from collections.abc import Iterable
from dataclasses import dataclass
import os
import signal
from typing import cast, override

from lightning import LightningDataModule, LightningModule
import torch
from torch.utils.data import TensorDataset
from torchdata.stateful_dataloader import StatefulDataLoader

from tslab.ml.interfaces import Batch
from tslab.ml.lightning import LightningTrainerConfig

# Three short epochs on the CPU; tests change single fields with dataclasses.replace.
FAST = LightningTrainerConfig(
    max_epochs=3, min_epochs=0, accelerator='cpu', devices=1, enable_progress_bar=False
)


@dataclass(frozen=True, kw_only=True)
class ToyInfo:
    num_features: int


class ToyData:
    """64 training and 16 validation samples of a noisy linear function, batches of 8."""

    def __init__(self, *, seed: int = 0, with_validation: bool = True) -> None:
        generator = torch.Generator().manual_seed(seed)
        weights = torch.randn(4, 1, generator=generator)
        inputs = torch.randn(80, 4, generator=generator)
        targets = inputs @ weights + 0.1 * torch.randn(80, 1, generator=generator)
        self._train = TensorDataset(inputs[:64], targets[:64])
        self._val = TensorDataset(inputs[64:], targets[64:])
        self._seed = seed
        self._with_validation = with_validation

    @property
    def info(self) -> ToyInfo:
        return ToyInfo(num_features=4)

    def train_loader(self) -> Iterable[Batch]:
        # StatefulDataLoader stores its shuffle state in Lightning checkpoints,
        # so a resumed training sees exactly the batches it would have seen.
        # torchdata types its batches as Any, so the loaders are cast to Batch.
        loader = StatefulDataLoader(
            dataset=self._train,
            batch_size=8,
            shuffle=True,
            generator=torch.Generator().manual_seed(self._seed),
        )
        return cast('Iterable[Batch]', loader)

    def val_loader(self) -> Iterable[Batch] | None:
        if not self._with_validation:
            return None
        return cast('Iterable[Batch]', StatefulDataLoader(dataset=self._val, batch_size=8))

    def test_loader(self) -> Iterable[Batch]:
        return cast('Iterable[Batch]', StatefulDataLoader(dataset=self._val, batch_size=8))


class ToyRegressor(LightningModule):
    """Linear regression with SGD and momentum, so the optimizer has state to restore."""

    def __init__(self, *, kill_at_step: int | None = None, fail_at_step: int | None = None) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(in_features=4, out_features=1)
        self.kill_at_step = kill_at_step
        self.fail_at_step = fail_at_step
        self.fit_starts = 0

    @override
    def on_fit_start(self) -> None:
        self.fit_starts += 1

    @override
    def training_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        if self.kill_at_step is not None and self.global_step == self.kill_at_step:
            # Like SLURM or the OOM killer: no cleanup, no checkpoint.
            os.kill(os.getpid(), signal.SIGKILL)
        if self.fail_at_step is not None and self.global_step == self.fail_at_step:
            message = 'CUDA out of memory (simulated)'
            raise RuntimeError(message)
        inputs, targets = batch
        loss = torch.nn.functional.mse_loss(input=self.layer(inputs), target=targets)
        self.log(name='train_loss', value=loss, on_step=True, on_epoch=True)
        return loss

    @override
    def validation_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        inputs, targets = batch
        loss = torch.nn.functional.mse_loss(input=self.layer(inputs), target=targets)
        self.log(name='val_loss', value=loss, on_step=True, on_epoch=True)
        return loss

    @override
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(params=self.parameters(), lr=0.05, momentum=0.9)


class ToyDataModule(LightningDataModule):
    """``ToyData`` as a datamodule without a validation split; records its hook calls."""

    def __init__(self) -> None:
        super().__init__()
        self.data = ToyData(with_validation=False)
        self.calls: list[str] = []

    @override
    def prepare_data(self) -> None:
        self.calls.append('prepare')

    @override
    def setup(self, stage: str) -> None:
        self.calls.append(f'setup:{stage}')

    @override
    def train_dataloader(self) -> Iterable[Batch]:
        return self.data.train_loader()

    @override
    def test_dataloader(self) -> Iterable[Batch]:
        return self.data.test_loader()

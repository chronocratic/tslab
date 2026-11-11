"""Fakes for the ml stage tests: no Lightning, all CPU."""

from collections.abc import Generator, Iterable
import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import override

import torch

from tslab.lab.core.run import Run, RunOptions
from tslab.ml.artifacts import Checkpoint
from tslab.ml.interfaces.data import Batch
from tslab.ml.interfaces.trainer import Trainer


@dataclass(frozen=True, kw_only=True)
class Config:
    seed: int = 0
    run: RunOptions = field(default_factory=RunOptions)


@contextlib.contextmanager
def run_in(*, root: Path, fail_fast: bool = True) -> Generator[Run, None, None]:
    config = Config(run=RunOptions(fail_fast=fail_fast))
    with Run(config=config, root=root, relative_path=Path('r')) as run:
        yield run


class FakeTrainer(Trainer[torch.nn.Module]):
    """Saves the untrained weights as the only checkpoint and counts calls."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.fit_calls = 0
        self.resumed = False

    def fingerprint_settings(self) -> str:
        return 'fake-trainer'

    @override
    def fit(
        self, *, run: Run, model: torch.nn.Module, data: object, checkpoint_directory: Path
    ) -> Checkpoint:
        self.fit_calls += 1
        last = checkpoint_directory / 'last.ckpt'
        self.resumed = last.is_file()
        torch.save(obj={'state_dict': model.state_dict()}, f=last)
        if self.fail:
            message = 'training time limit reached (simulated)'
            raise RuntimeError(message)
        return Checkpoint(directory=checkpoint_directory, best='last.ckpt', last='last.ckpt')

    @override
    def load(self, *, model: torch.nn.Module, checkpoint: Checkpoint) -> None:
        state = torch.load(f=checkpoint.best_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict=state['state_dict'])


class LinearModel(torch.nn.Module):
    """A single linear layer that can be fingerprinted."""

    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(in_features=10, out_features=4)

    def fingerprint_settings(self) -> str:
        return 'linear-model'


class EmptyData:
    """A data source whose loaders are never read: the fake trainer ignores them."""

    @property
    def info(self) -> None:
        return None

    def fingerprint_settings(self) -> str:
        return 'empty-data'

    def train_loader(self) -> Iterable[Batch]:
        batches: list[Batch] = []
        return batches

    def val_loader(self) -> Iterable[Batch] | None:
        return None

    def test_loader(self) -> Iterable[Batch]:
        batches: list[Batch] = []
        return batches

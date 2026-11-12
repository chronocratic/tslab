import dataclasses
import os
from pathlib import Path
import signal
from typing import override

import pytest
import torch

from tests.ml.integration.lightning_toy import FAST, ToyData, ToyRegressor
from tests.support.integration_helpers import run_in
import tslab.lab
from tslab.lab.core.status import read_status
from tslab.lab.enums import RunState
import tslab.ml
from tslab.ml.lightning import LightningTrainer
from tslab.ml.stages.train import train as train_stage


class _SignaledRegressor(ToyRegressor):
    def fingerprint_settings(self) -> str:
        return 'signaled-toy-regressor'

    @override
    def training_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = super().training_step(batch, batch_idx)
        if self.global_step == 3:
            os.kill(os.getpid(), signal.SIGUSR2)
        return loss


class _SignaledData(ToyData):
    def fingerprint_settings(self) -> str:
        return 'signaled-toy-data'


def _run_signaled(*, root: Path, min_epochs: int = 0) -> None:
    config = dataclasses.replace(FAST, max_epochs=60 if min_epochs else 3, min_epochs=min_epochs)
    with run_in(root=root) as run:
        train_stage(
            run=run,
            trainer=LightningTrainer(config=config),
            model=_SignaledRegressor(),
            data=_SignaledData(),
        )


@pytest.mark.slow
def test_usr2_during_fit_saves_last_and_resumes(tmp_path: Path) -> None:
    with pytest.raises(tslab.lab.JobTimeLimitSignal):
        _run_signaled(root=tmp_path)
    checkpoint = tmp_path / 'r' / 'train' / 'default' / 'last.ckpt'
    assert checkpoint.is_file()
    assert torch.load(checkpoint, weights_only=False)['global_step'] >= 3
    snapshot = read_status(path=tmp_path / 'r' / 'status.json')
    assert snapshot is not None
    error = snapshot.instances['train[default]'].error
    assert error is not None
    assert error.startswith('JobTimeLimitSignal:')
    _run_signaled(root=tmp_path)
    finished = read_status(path=tmp_path / 'r' / 'status.json')
    assert finished is not None
    assert finished.run_state is RunState.FINISHED


@pytest.mark.slow
def test_usr2_with_min_epochs_still_stops(tmp_path: Path) -> None:
    with pytest.raises(tslab.lab.JobTimeLimitSignal):
        _run_signaled(root=tmp_path, min_epochs=50)
    assert (tmp_path / 'r' / 'train' / 'default' / 'last.ckpt').is_file()

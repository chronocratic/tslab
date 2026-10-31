import dataclasses
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import override

from integration_helpers import run_in
from lightning_toy import FAST, ToyData, ToyRegressor
import pytest
from safetensors.torch import load_file
import torch

import tslab
from tslab.core.status import read_status, RunState
from tslab.lightning import LightningTrainer
from tslab.stages import train as train_stage

SCRIPT = Path(__file__).parent / 'lightning_resume_script.py'


class _SignalledRegressor(ToyRegressor):
    def __init__(self, *, send_term: bool = False) -> None:
        super().__init__()
        self.send_term = send_term

    def fingerprint_settings(self) -> dict[str, bool]:
        return {'send_term': self.send_term}

    @override
    def training_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = super().training_step(batch, batch_idx)
        if self.global_step == 3:
            os.kill(os.getpid(), signal.SIGUSR2)
            if self.send_term:
                os.kill(os.getpid(), signal.SIGTERM)
        return loss


class _SignalledData(ToyData):
    def fingerprint_settings(self) -> str:
        return 'signalled-toy-data'


def _run_signalled(*, root: Path, send_term: bool = False, min_epochs: int = 0) -> None:
    config = dataclasses.replace(FAST, max_epochs=60 if min_epochs else 3, min_epochs=min_epochs)
    with run_in(root=root) as run:
        train_stage(
            run=run,
            trainer=LightningTrainer(config=config),
            model=_SignalledRegressor(send_term=send_term),
            data=_SignalledData(),
        )


@pytest.mark.slow
def test_usr2_during_fit_saves_last_and_resumes(tmp_path: Path) -> None:
    with pytest.raises(tslab.JobTimeLimitSignal):
        _run_signalled(root=tmp_path)
    checkpoint = tmp_path / 'r' / 'train' / 'default' / 'last.ckpt'
    assert checkpoint.is_file()
    assert torch.load(checkpoint, weights_only=False)['global_step'] >= 3
    snapshot = read_status(path=tmp_path / 'r' / 'status.json')
    assert snapshot is not None
    error = snapshot.instances['train[default]'].error
    assert error is not None
    assert error.startswith('JobTimeLimitSignal:')
    _run_signalled(root=tmp_path)
    finished = read_status(path=tmp_path / 'r' / 'status.json')
    assert finished is not None
    assert finished.run_state is RunState.FINISHED


@pytest.mark.slow
def test_usr2_then_sigterm_during_fit_still_saves(tmp_path: Path) -> None:
    with pytest.raises(tslab.JobTimeLimitSignal):
        _run_signalled(root=tmp_path, send_term=True)
    assert (tmp_path / 'r' / 'train' / 'default' / 'last.ckpt').is_file()


@pytest.mark.slow
def test_usr2_with_min_epochs_still_stops(tmp_path: Path) -> None:
    with pytest.raises(tslab.JobTimeLimitSignal):
        _run_signalled(root=tmp_path, min_epochs=50)
    assert (tmp_path / 'r' / 'train' / 'default' / 'last.ckpt').is_file()


def train(*, root: Path, kill_at_step: int | None = None) -> int:
    arguments = [sys.executable, str(SCRIPT), str(root)]
    if kill_at_step is not None:
        arguments.append(str(kill_at_step))
    result = subprocess.run(  # noqa: S603 # why: runs this repository's own script
        arguments, capture_output=True, text=True, check=False, cwd=SCRIPT.parent
    )
    if result.returncode not in {0, -signal.SIGKILL}:
        pytest.fail(f'training script failed:\n{result.stderr}')
    return result.returncode


def finite_weights(path: Path) -> dict[str, torch.Tensor]:
    tensors = load_file(filename=path)
    for name, tensor in tensors.items():
        assert torch.isfinite(tensor).all(), name
    return tensors


def killed_resume_cycle(root: Path) -> dict[str, torch.Tensor]:
    # Step 12 is the middle of the second epoch (8 batches per epoch).
    assert train(root=root, kill_at_step=12) == -signal.SIGKILL
    assert not (root / 'weights.safetensors').exists()
    assert (root / 'toy' / 'train' / 'last.ckpt').is_file()
    assert train(root=root) == 0
    return finite_weights(root / 'weights.safetensors')


@pytest.mark.slow
def test_killed_training_resumes_deterministically(tmp_path: Path) -> None:
    # A kill mid-epoch loses the in-epoch batches; Lightning 2.6.6 does not apply the
    # saved StatefulDataLoader state on an epoch-end resume, so the redone epoch uses a
    # fresh (seeded, hence deterministic) batch order and the final weights can differ
    # from an uninterrupted run. Weights and optimizer state ARE restored bit-identically
    # at the resume point; what this test pins is that two identical kill+resume cycles
    # end bit-identical. If bit-identical resume is required, seed the per-epoch shuffle
    # from the epoch number instead.
    first = killed_resume_cycle(root=tmp_path / 'a')
    second = killed_resume_cycle(root=tmp_path / 'b')
    assert first.keys() == second.keys()
    for name, tensor in first.items():
        assert tensor.equal(second[name]), name

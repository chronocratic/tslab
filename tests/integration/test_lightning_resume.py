from pathlib import Path
import signal
import subprocess
import sys

import pytest
from safetensors.torch import load_file
import torch

SCRIPT = Path(__file__).parent / 'lightning_resume_script.py'


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
    # from the epoch number instead (see the M4 carry-forward in the plan ledger).
    first = killed_resume_cycle(root=tmp_path / 'a')
    second = killed_resume_cycle(root=tmp_path / 'b')
    assert first.keys() == second.keys()
    for name, tensor in first.items():
        assert tensor.equal(second[name]), name

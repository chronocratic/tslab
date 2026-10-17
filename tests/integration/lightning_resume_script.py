"""Train the toy regressor inside a run; optionally SIGKILL itself at a global step.

Usage: python lightning_resume_script.py ROOT [KILL_AT_STEP]
After a complete training it writes ROOT/weights.safetensors.
"""

import dataclasses
from dataclasses import dataclass
from pathlib import Path
import sys

from lightning_toy import FAST, ToyData, ToyRegressor
from safetensors.torch import save_file
import torch

from tslab.core.run import Run
from tslab.lightning import LightningTrainer


@dataclass(frozen=True, kw_only=True)
class ScriptConfig:
    seed: int = 0


def main(*, root: Path, kill_at_step: int | None) -> None:
    torch.use_deterministic_algorithms(mode=True)
    config = dataclasses.replace(FAST, max_epochs=4, extra_trainer_kwargs={'deterministic': True})
    with Run(config=ScriptConfig(), root=root, relative_path=Path('toy')) as run:
        model = ToyRegressor(kill_at_step=kill_at_step)
        LightningTrainer(config=config).fit(
            run=run, model=model, data=ToyData(), checkpoint_directory=run.directory / 'train'
        )
    weights = {name: tensor.contiguous() for name, tensor in model.state_dict().items()}
    save_file(tensors=weights, filename=root / 'weights.safetensors')


if __name__ == '__main__':
    main(
        root=Path(sys.argv[1]),
        # why: argv index
        kill_at_step=int(sys.argv[2]) if len(sys.argv) > 2 else None,
    )

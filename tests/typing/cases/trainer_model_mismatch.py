# expect-error: invalid-argument-type
from pathlib import Path

import torch

import tslab
from tslab.interfaces import DataSource, Trainer


def fit_with_a_string_model(
    *, trainer: Trainer[torch.nn.Module], run: tslab.Run, data: DataSource[object]
) -> tslab.Checkpoint:
    return trainer.fit(
        run=run, model='not a module', data=data, checkpoint_directory=Path('checkpoints')
    )

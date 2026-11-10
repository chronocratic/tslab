# expect-error: invalid-argument-type
import torch

import tslab
from tslab.interfaces import DataSource, Trainer


def train_a_string(
    *, run: tslab.Run, trainer: Trainer[torch.nn.Module], data: DataSource[object]
) -> None:
    tslab.stages.train(run=run, trainer=trainer, model='not a module', data=data)

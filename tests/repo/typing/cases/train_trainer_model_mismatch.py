# expect-error: invalid-argument-type
import torch

import tslab.lab
import tslab.ml
from tslab.ml.interfaces import DataSource, Trainer


def train_a_string(
    *, run: tslab.lab.Run, trainer: Trainer[torch.nn.Module], data: DataSource[object]
) -> None:
    tslab.ml.stages.train(run=run, trainer=trainer, model='not a module', data=data)

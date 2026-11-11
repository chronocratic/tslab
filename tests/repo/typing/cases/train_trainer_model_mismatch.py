# expect-error: invalid-argument-type
import torch

import tslab.lab
import tslab.ml
import tslab.ml.stages
from tslab.ml.interfaces.data import DataSource
from tslab.ml.interfaces.trainer import Trainer


def train_a_string(
    *, run: tslab.lab.Run, trainer: Trainer[torch.nn.Module], data: DataSource[object]
) -> None:
    tslab.ml.stages.train(run=run, trainer=trainer, model='not a module', data=data)

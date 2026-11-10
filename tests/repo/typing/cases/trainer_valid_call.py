# expect-error: none
from pathlib import Path

import torch

import tslab.lab
import tslab.ml
from tslab.ml.interfaces import DataSource, Trainer


def fit_a_linear_model(
    *, trainer: Trainer[torch.nn.Module], run: tslab.lab.Run, data: DataSource[object]
) -> torch.nn.Linear:
    model = torch.nn.Linear(in_features=4, out_features=1)
    checkpoint = trainer.fit(
        run=run, model=model, data=data, checkpoint_directory=Path('checkpoints')
    )
    trainer.load(model=model, checkpoint=checkpoint)
    return model


def a_general_trainer_accepts_a_narrower_model(
    *, trainer: Trainer[torch.nn.Module], narrow: Trainer[torch.nn.Linear]
) -> Trainer[torch.nn.Linear]:
    # M only appears as an argument type, so a trainer for any module is also a
    # trainer for linear layers. The ready encode stages rely on this.
    del narrow
    return trainer

# expect-error: none
from typing import cast

import torch

import tslab.lab
import tslab.ml
import tslab.ts
import tslab.ts.stages
from tslab.ts.interfaces.encoder import Encoder
from tslab.ml.interfaces.data import LabeledSplits
from tslab.ml.interfaces.trainer import Trainer


class LinearModel(torch.nn.Linear):
    def encode(self, *, inputs: torch.Tensor) -> torch.Tensor:
        # why: torch ships no stub for Module.__call__, so the call is Any
        return cast('torch.Tensor', self(inputs))


def linear_encoder(model: LinearModel) -> Encoder:
    return model


def encode_with_a_general_trainer(
    *,
    run: tslab.lab.Run,
    trainer: Trainer[torch.nn.Module],
    checkpoint: tslab.ml.Checkpoint,
    data: LabeledSplits,
) -> tslab.ml.Tensors:
    return tslab.ts.stages.encode(
        run=run,
        trainer=trainer,
        model=LinearModel(in_features=2, out_features=2),
        checkpoint=checkpoint,
        data=data,
        encoder=linear_encoder,
    )

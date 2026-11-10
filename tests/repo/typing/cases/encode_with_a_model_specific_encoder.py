# expect-error: none
from typing import cast

import torch

import tslab.lab
import tslab.ml
import tslab.ts
from tslab.ts.interfaces import Encoder
from tslab.ml.interfaces import LabeledSplits, Trainer


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
    checkpoint: tslab.lab.Checkpoint,
    data: LabeledSplits,
) -> tslab.lab.Tensors:
    return tslab.ts.stages.encode(
        run=run,
        trainer=trainer,
        model=LinearModel(in_features=2, out_features=2),
        checkpoint=checkpoint,
        data=data,
        encoder=linear_encoder,
    )

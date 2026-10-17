# expect-error: none
from typing import cast

import torch

import tslab
from tslab.interfaces import Encoder, LabeledSplits, Trainer


class LinearModel(torch.nn.Linear):
    def encode(self, *, inputs: torch.Tensor) -> torch.Tensor:
        # why: torch ships no stub for Module.__call__, so the call is Any
        return cast('torch.Tensor', self(inputs))


def linear_encoder(model: LinearModel) -> Encoder:
    return model


def encode_with_a_general_trainer(
    *,
    run: tslab.Run,
    trainer: Trainer[torch.nn.Module],
    checkpoint: tslab.Checkpoint,
    data: LabeledSplits,
) -> tslab.Tensors:
    return tslab.stages.encode(
        run=run,
        trainer=trainer,
        model=LinearModel(in_features=2, out_features=2),
        checkpoint=checkpoint,
        data=data,
        encoder=linear_encoder,
    )

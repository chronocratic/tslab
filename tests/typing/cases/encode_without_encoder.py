# expect-error: missing-argument
import torch

import tslab
from tslab.interfaces import LabeledSplits, Trainer


def encode_without_encoder(
    *,
    run: tslab.Run,
    trainer: Trainer[torch.nn.Module],
    checkpoint: tslab.Checkpoint,
    data: LabeledSplits,
) -> tslab.Tensors:
    return tslab.stages.encode(
        run=run,
        trainer=trainer,
        model=torch.nn.Linear(in_features=2, out_features=2),
        checkpoint=checkpoint,
        data=data,
    )

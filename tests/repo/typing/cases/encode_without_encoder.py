# expect-error: missing-argument
import torch

import tslab.lab
import tslab.ml
import tslab.ts
from tslab.ml.interfaces import LabeledSplits, Trainer


def encode_without_encoder(
    *,
    run: tslab.lab.Run,
    trainer: Trainer[torch.nn.Module],
    checkpoint: tslab.ml.Checkpoint,
    data: LabeledSplits,
) -> tslab.ml.Tensors:
    return tslab.ts.stages.encode(
        run=run,
        trainer=trainer,
        model=torch.nn.Linear(in_features=2, out_features=2),
        checkpoint=checkpoint,
        data=data,
    )

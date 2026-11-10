"""Shared helpers of the stage tests."""

from tests.ml.unit.stage_fakes import FakeTrainer
from tests.ts.unit.stage_fakes import LinearEncoder, ToySplits
import tslab.lab
import tslab.ml
import tslab.ts

__all__ = ['encoded']


def encoded(*, run: tslab.lab.Run, model: LinearEncoder | None = None) -> tslab.ml.Tensors:
    model = LinearEncoder() if model is None else model
    trainer = FakeTrainer()
    checkpoint = tslab.ml.stages.train(run=run, trainer=trainer, model=model, data=ToySplits())
    return tslab.ts.stages.encode(
        run=run,
        trainer=trainer,
        model=model,
        checkpoint=checkpoint,
        data=ToySplits(),
        encoder=tslab.ts.ModelAsEncoder(device='cpu'),
    )

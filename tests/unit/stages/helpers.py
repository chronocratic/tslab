"""Shared helpers of the stage tests."""

from tests.unit.stage_fakes import FakeTrainer, LinearEncoder, ToySplits
import tslab

__all__ = ['encoded']


def encoded(*, run: tslab.Run, model: LinearEncoder | None = None) -> tslab.Tensors:
    model = LinearEncoder() if model is None else model
    trainer = FakeTrainer()
    checkpoint = tslab.stages.train(run=run, trainer=trainer, model=model, data=ToySplits())
    return tslab.stages.encode(
        run=run,
        trainer=trainer,
        model=model,
        checkpoint=checkpoint,
        data=ToySplits(),
        encoder=tslab.ModelAsEncoder(device='cpu'),
    )

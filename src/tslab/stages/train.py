"""The ready ``train`` stage."""

from tslab.core.artifact import Checkpoint
from tslab.core.paths import InstanceName
from tslab.core.run import Run
from tslab.core.stage import stage
from tslab.interfaces import DataSource, Trainer

__all__ = ['train']

STAGE_NAME = 'train'


@stage(name=STAGE_NAME, output=Checkpoint, in_place=True)
def train[M](
    *,
    run: Run,
    name: InstanceName | None = None,
    trainer: Trainer[M],
    model: M,
    data: DataSource[object],
) -> Checkpoint:
    """Train ``model`` and return its best and last checkpoints.

    The trainer writes checkpoints straight into the folder of this instance. A
    training interrupted by a crash, a kill or a time limit therefore resumes
    from its last checkpoint the next time the script runs with the same
    settings.

    Args:
        run: The current run.
        name: Instance name, for training several models in one run.
        trainer: The trainer, for example ``tslab.lightning.LightningTrainer``.
        model: The model; trained in place. It must be fingerprintable.
        data: The training data.

    Returns:
        The checkpoint naming the best and the last weights.
    """
    return trainer.fit(
        run=run,
        model=model,
        data=data,
        checkpoint_directory=run.instance_directory(stage_name=STAGE_NAME, instance_name=name),
    )

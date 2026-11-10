"""The trainer interface: fit a model on a data source, then load its best weights."""

from abc import ABC, abstractmethod
from pathlib import Path

from tslab.lab.core.run import Run
from tslab.ml.artifacts import Checkpoint
from tslab.ml.interfaces.data import DataSource

__all__ = ['Trainer']


class Trainer[M](ABC):
    """Trains models of type ``M`` and loads their weights.

    ``M`` ties a trainer to the models it understands, so a type checker
    rejects passing a model the trainer cannot train. A trainer is also an
    argument of the ready ``train`` stage, so implementations provide
    ``fingerprint_settings()`` (see ``tslab.lab.Fingerprintable``).
    """

    @abstractmethod
    def fit(
        self, *, run: Run, model: M, data: DataSource[object], checkpoint_directory: Path
    ) -> Checkpoint:
        """Train ``model``, resuming from checkpoints already in ``checkpoint_directory``.

        Args:
            run: The current run; its trackers receive the training metrics.
            model: The model to train, updated in place.
            data: The training and validation data.
            checkpoint_directory: Folder for the checkpoint files; it may hold
                checkpoints of an interrupted earlier attempt.

        Returns:
            The checkpoint naming the best and the last weights.
        """

    @abstractmethod
    def load(self, *, model: M, checkpoint: Checkpoint) -> None:
        """Load the best weights of ``checkpoint`` into ``model``, in place.

        Nothing is returned, so ``M`` only appears as an argument type. That
        makes ``Trainer`` contravariant in ``M``: a trainer for any
        ``LightningModule`` is also accepted where a trainer for one model class
        is expected, which lets a stage pair it with an encoder written for
        that class.

        Args:
            model: A model of the same architecture as the trained one.
            checkpoint: The checkpoint returned by ``fit`` (or an external one).
        """

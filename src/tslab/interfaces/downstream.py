"""Downstream models: fitted on representations, then used to predict."""

from abc import ABC, abstractmethod

import torch

__all__ = ['DownstreamModel', 'Probe']


class DownstreamModel(ABC):
    """A model fitted on inputs (usually representations) that predicts targets."""

    @abstractmethod
    def fit(
        self,
        *,
        train_inputs: torch.Tensor,
        train_targets: torch.Tensor,
        val_inputs: torch.Tensor | None,
        val_targets: torch.Tensor | None,
    ) -> None:
        """Fit the model; the validation split, if given, selects hyperparameters.

        Args:
            train_inputs: Training inputs, one row per sample.
            train_targets: Training targets.
            val_inputs: Validation inputs, or ``None``.
            val_targets: Validation targets, or ``None``.
        """

    @abstractmethod
    def predict(self, *, inputs: torch.Tensor) -> torch.Tensor:
        """Predict targets for ``inputs``.

        Args:
            inputs: One row per sample.

        Returns:
            One prediction per row.
        """

    def default_metrics(
        self, *, predictions: torch.Tensor, targets: torch.Tensor
    ) -> dict[str, float]:
        """Return the metrics of this model's evaluation protocol.

        ``evaluate`` uses them when it is called without ``metrics=``.

        Raises:
            NotImplementedError: Unless a subclass defines its protocol metrics.
        """
        del predictions, targets
        message = (
            f'{type(self).__qualname__} has no protocol metrics; pass metrics= to evaluate '
            'or override default_metrics'
        )
        raise NotImplementedError(message)


class Probe(DownstreamModel, ABC):
    """A downstream model trained on frozen representations.

    The encoder is not updated; the probe measures how much the
    representations already contain. A model trained together with the
    encoder would be a head (fine-tuning), which TSLab does not provide yet.
    """

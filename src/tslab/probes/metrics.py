"""The metrics of the evaluation protocols."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

__all__ = ['classification_metrics', 'forecasting_metrics']

_BINARY_CLASS_COUNT = 2
# Lower bound of |target| in the MAPE denominator, so zero targets stay finite.
_MAPE_MIN_DENOMINATOR = 1e-8


def classification_metrics(*, predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Return accuracy, precision, recall and F1.

    Precision, recall and F1 use ``average='binary'`` when the targets contain
    exactly two classes, and ``'macro'`` otherwise.

    Args:
        predictions: Predicted labels.
        targets: True labels.

    Returns:
        The metrics keyed ``accuracy``, ``precision``, ``recall``, ``f1``.
    """
    average = 'binary' if len(np.unique(targets)) == _BINARY_CLASS_COUNT else 'macro'
    return {
        'accuracy': float(accuracy_score(y_true=targets, y_pred=predictions)),
        'precision': float(precision_score(y_true=targets, y_pred=predictions, average=average)),
        'recall': float(recall_score(y_true=targets, y_pred=predictions, average=average)),
        'f1': float(f1_score(y_true=targets, y_pred=predictions, average=average)),
    }


def forecasting_metrics(*, predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Return mean squared, mean absolute and mean absolute percentage error.

    Args:
        predictions: Predicted values.
        targets: True values, same shape.

    Returns:
        The metrics keyed ``mse_loss``, ``mae_loss``, ``mape_loss``.
    """
    return {
        'mse_loss': float(np.mean((predictions - targets) ** 2)),
        'mae_loss': float(np.mean(np.abs(predictions - targets))),
        'mape_loss': float(
            np.mean(np.abs(predictions - targets) / np.abs(targets).clip(min=_MAPE_MIN_DENOMINATOR))
        ),
    }

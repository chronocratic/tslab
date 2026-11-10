"""The metrics of the evaluation protocols."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

__all__ = ['classification_metrics', 'forecasting_metrics']

_BINARY_CLASS_COUNT = 2
# Lower bound of |target| in the MAPE denominator, so zero targets stay finite.
_MAPE_MIN_DENOMINATOR = 1e-8


def _require_same_shape(*, predictions: np.ndarray, targets: np.ndarray) -> None:
    if predictions.shape != targets.shape:
        message = (
            f'predictions have shape {predictions.shape} but targets have shape {targets.shape}'
        )
        raise ValueError(message)


def classification_metrics(
    *, predictions: np.ndarray, targets: np.ndarray, classes: np.ndarray | None = None
) -> dict[str, float]:
    """Return accuracy, precision, recall and F1.

    Precision, recall and F1 use ``average='binary'`` when the problem has
    exactly two classes, and ``'macro'`` otherwise. Classes with no predicted
    or true sample score 0 instead of warning.

    Args:
        predictions: Predicted labels.
        targets: True labels.
        classes: The classes the model was fitted on. Decides binary or macro,
            so a test split that holds fewer classes does not change the
            average. ``None`` counts the classes in ``targets``.

    Returns:
        The metrics keyed ``accuracy``, ``precision``, ``recall``, ``f1``.
    """
    _require_same_shape(predictions=predictions, targets=targets)
    class_count = len(np.unique(targets if classes is None else classes))
    average = 'binary' if class_count == _BINARY_CLASS_COUNT else 'macro'
    scores = {'precision': precision_score, 'recall': recall_score, 'f1': f1_score}
    return {
        'accuracy': float(accuracy_score(y_true=targets, y_pred=predictions)),
        **{
            name: float(score(y_true=targets, y_pred=predictions, average=average, zero_division=0))
            for name, score in scores.items()
        },
    }


def forecasting_metrics(*, predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Return mean squared, mean absolute and mean absolute percentage error.

    Args:
        predictions: Predicted values.
        targets: True values, same shape.

    Returns:
        The metrics keyed ``mse_loss``, ``mae_loss``, ``mape_loss``.

    Raises:
        ValueError: If the shapes differ (they would broadcast to a wrong mean).
    """
    _require_same_shape(predictions=predictions, targets=targets)
    return {
        'mse_loss': float(np.mean((predictions - targets) ** 2)),
        'mae_loss': float(np.mean(np.abs(predictions - targets))),
        'mape_loss': float(
            np.mean(np.abs(predictions - targets) / np.abs(targets).clip(min=_MAPE_MIN_DENOMINATOR))
        ),
    }

import numpy as np
import pytest
from sklearn.metrics import f1_score, precision_score

from tslab.ts.probes import classification_metrics, forecasting_metrics


@pytest.mark.parametrize('classes', [2, 3])
def test_classification_metrics_average_binary_or_macro(*, classes: int) -> None:
    generator = np.random.default_rng(seed=0)
    targets = generator.integers(low=0, high=classes, size=60)
    predictions = generator.integers(low=0, high=classes, size=60)
    average = 'binary' if classes == 2 else 'macro'
    metrics = classification_metrics(predictions=predictions, targets=targets)
    assert metrics['f1'] == f1_score(y_true=targets, y_pred=predictions, average=average)
    assert metrics['precision'] == precision_score(
        y_true=targets, y_pred=predictions, average=average
    )
    assert set(metrics) == {'accuracy', 'precision', 'recall', 'f1'}


def test_the_average_follows_the_fitted_classes_not_the_test_labels() -> None:
    # Three classes were fitted, but the test split only holds two of them.
    targets = np.array([0, 0, 1, 1])
    predictions = np.array([0, 2, 1, 1])
    metrics = classification_metrics(
        predictions=predictions, targets=targets, classes=np.array([0, 1, 2])
    )
    assert metrics['f1'] == f1_score(y_true=targets, y_pred=predictions, average='macro')


def test_metrics_reject_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match='shape'):
        forecasting_metrics(predictions=np.zeros((4, 1)), targets=np.zeros(4))
    with pytest.raises(ValueError, match='shape'):
        classification_metrics(predictions=np.zeros((4, 1)), targets=np.zeros(4))


def test_mape_divides_by_at_least_1e_minus_8() -> None:
    metrics = forecasting_metrics(predictions=np.array([1.0, 2.0]), targets=np.array([0.0, 2.0]))
    assert metrics == pytest.approx({'mse_loss': 0.5, 'mae_loss': 0.5, 'mape_loss': 0.5 / 1e-8})


@pytest.mark.parametrize('labels', [(0, 2), ('a', 'b')])
def test_binary_metrics_accept_labels_without_one(labels: tuple[object, object]) -> None:
    targets = np.array([labels[0], labels[1], labels[1], labels[0]])
    scores = classification_metrics(predictions=targets, targets=targets)
    assert scores['f1'] == 1.0

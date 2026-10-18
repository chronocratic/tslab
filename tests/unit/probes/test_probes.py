import random

import numpy as np
import pytest
from sklearn.metrics import f1_score, precision_score
import torch

from tslab.core.fingerprint import canonicalize
from tslab.probes import classification_metrics, forecasting_metrics, RidgeProbe, SVMProbe


def blobs(*, per_class: int, classes: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(classes).repeat_interleave(per_class)
    centers = 3.0 * labels.to(dtype=torch.float32)[:, None] * torch.ones(1, 4)
    return centers + 0.3 * torch.randn(len(labels), 4, generator=generator), labels


def fitted_svm(*, train: int, val: int, classes: int = 2, **settings: int) -> SVMProbe:
    train_inputs, train_targets = blobs(per_class=train, classes=classes, seed=0)
    val_inputs, val_targets = blobs(per_class=val, classes=classes, seed=1)
    probe = SVMProbe(n_jobs=1, **settings)
    probe.fit(
        train_inputs=train_inputs,
        train_targets=train_targets,
        val_inputs=val_inputs,
        val_targets=val_targets,
    )
    return probe


def test_small_datasets_use_a_near_hard_margin_svc() -> None:
    probe = fitted_svm(train=3, val=1, classes=3)
    svc = probe.pipeline.named_steps['svc']
    assert svc.C == 1e6
    assert svc.shape_fit_[0] == 12


def test_larger_datasets_search_c_and_refit_on_train_and_validation() -> None:
    probe = fitted_svm(train=30, val=10)
    svc = probe.pipeline.named_steps['svc']
    assert svc.C in {1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1e3, 1e4, 1e6}
    assert svc.shape_fit_[0] == 80
    test_inputs, test_targets = blobs(per_class=10, classes=2, seed=2)
    assert torch.equal(probe.predict(inputs=test_inputs), test_targets)


def test_the_svm_caps_the_training_split() -> None:
    probe = fitted_svm(train=30, val=10, max_train_samples=20)
    assert probe.pipeline.named_steps['svc'].shape_fit_[0] == 40


def test_collapsed_embeddings_stay_finite() -> None:
    # Values 0 and the smallest float32 subnormal: the float32 scale underflows to 0.
    tiny = torch.tensor(1e-45, dtype=torch.float32)
    collapsed = (torch.arange(80) % 2).to(dtype=torch.float32)[:, None] * tiny * torch.ones(1, 4)
    labels = torch.arange(80) % 2
    probe = SVMProbe(n_jobs=1)
    probe.fit(
        train_inputs=collapsed[:60],
        train_targets=labels[:60],
        val_inputs=collapsed[60:],
        val_targets=labels[60:],
    )
    transformed = probe.pipeline[:-1].transform(collapsed.numpy())
    assert transformed.dtype == np.float64
    assert np.isfinite(transformed).all()


def ridge_data(*, val_is_zero: bool) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    inputs = torch.randn(80, 5, generator=generator)
    targets = inputs @ torch.randn(5, 2, generator=generator)
    val_targets = torch.zeros(20, 2) if val_is_zero else targets[60:]
    return {
        'train_inputs': inputs[:60],
        'train_targets': targets[:60],
        'val_inputs': inputs[60:],
        'val_targets': val_targets,
    }


@pytest.mark.parametrize(('val_is_zero', 'alpha'), [(True, 1000), (False, 0.1)])
def test_the_ridge_alpha_is_chosen_on_the_validation_split(
    *, val_is_zero: bool, alpha: float
) -> None:
    probe = RidgeProbe(n_jobs=1)
    probe.fit(**ridge_data(val_is_zero=val_is_zero))
    assert probe.pipeline.named_steps['ridge'].alpha == alpha


def test_the_ridge_caps_train_and_validation() -> None:
    probe = RidgeProbe(max_samples=10, n_jobs=1)
    probe.fit(**ridge_data(val_is_zero=False))
    assert probe.pipeline.named_steps['scaler'].n_samples_seen_ == 20


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


def test_svm_default_metrics_use_the_fitted_classes() -> None:
    probe = fitted_svm(train=30, val=10, classes=3)
    targets = torch.tensor([0, 0, 1, 1])
    predictions = torch.tensor([0, 2, 1, 1])
    metrics = probe.default_metrics(predictions=predictions, targets=targets)
    assert metrics['f1'] == f1_score(
        y_true=targets.numpy(), y_pred=predictions.numpy(), average='macro'
    )


def test_a_single_class_or_empty_validation_split_gets_a_clear_error() -> None:
    inputs = torch.randn(60, 4)
    with pytest.raises(ValueError, match='at least 2 classes'):
        SVMProbe(n_jobs=1).fit(
            train_inputs=inputs,
            train_targets=torch.zeros(60, dtype=torch.long),
            val_inputs=inputs[:10],
            val_targets=torch.zeros(10, dtype=torch.long),
        )
    with pytest.raises(ValueError, match='empty'):
        RidgeProbe(n_jobs=1).fit(
            train_inputs=inputs, train_targets=inputs, val_inputs=inputs[:0], val_targets=inputs[:0]
        )


def test_the_cap_survives_a_class_with_one_member() -> None:
    inputs, labels = blobs(per_class=30, classes=2, seed=0)
    labels[0] = 5
    probe = SVMProbe(max_train_samples=20, n_jobs=1)
    probe.fit(
        train_inputs=inputs, train_targets=labels, val_inputs=inputs[:10], val_targets=labels[:10]
    )
    assert probe.pipeline.named_steps['svc'].shape_fit_[0] == 30


def test_metrics_reject_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match='shape'):
        forecasting_metrics(predictions=np.zeros((4, 1)), targets=np.zeros(4))
    with pytest.raises(ValueError, match='shape'):
        classification_metrics(predictions=np.zeros((4, 1)), targets=np.zeros(4))


def test_mape_divides_by_at_least_1e_minus_8() -> None:
    metrics = forecasting_metrics(predictions=np.array([1.0, 2.0]), targets=np.array([0.0, 2.0]))
    assert metrics == pytest.approx({'mse_loss': 0.5, 'mae_loss': 0.5, 'mape_loss': 0.5 / 1e-8})


def test_default_metrics_use_the_protocol_names() -> None:
    probe = fitted_svm(train=30, val=10)
    inputs, targets = blobs(per_class=10, classes=2, seed=2)
    predictions = probe.predict(inputs=inputs)
    assert probe.default_metrics(predictions=predictions, targets=targets)['accuracy'] == 1.0


def test_fitting_leaves_the_global_random_state_alone() -> None:
    python_state = random.getstate()
    numpy_key = np.random.get_state(legacy=False)['state']['key'].copy()  # noqa: NPY002
    torch_state = torch.get_rng_state()
    fitted_svm(train=30, val=10)
    RidgeProbe(n_jobs=1).fit(**ridge_data(val_is_zero=False))
    assert random.getstate() == python_state
    assert np.array_equal(
        np.random.get_state(legacy=False)['state']['key'],  # noqa: NPY002
        numpy_key,
    )
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_the_validation_split_is_required() -> None:
    inputs, targets = blobs(per_class=30, classes=2, seed=0)
    with pytest.raises(ValueError, match='validation'):
        SVMProbe().fit(
            train_inputs=inputs, train_targets=targets, val_inputs=None, val_targets=None
        )


def test_predicting_before_fitting_raises() -> None:
    with pytest.raises(RuntimeError, match='fit'):
        RidgeProbe().predict(inputs=torch.zeros(2, 3))


def test_caps_are_fingerprinted_and_worker_counts_are_not() -> None:
    assert canonicalize(value=SVMProbe(n_jobs=1)) == canonicalize(value=SVMProbe(n_jobs=4))
    assert canonicalize(value=SVMProbe()) != canonicalize(value=SVMProbe(max_train_samples=None))
    assert canonicalize(value=RidgeProbe()) != canonicalize(value=SVMProbe())

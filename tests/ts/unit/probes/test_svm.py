import numpy as np
from sklearn.metrics import f1_score
import torch

from tests.unit.probes.helpers import blobs, fitted_svm
from tslab.probes import SVMProbe


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


def test_svm_default_metrics_use_the_fitted_classes() -> None:
    probe = fitted_svm(train=30, val=10, classes=3)
    targets = torch.tensor([0, 0, 1, 1])
    predictions = torch.tensor([0, 2, 1, 1])
    metrics = probe.default_metrics(predictions=predictions, targets=targets)
    assert metrics['f1'] == f1_score(
        y_true=targets.numpy(), y_pred=predictions.numpy(), average='macro'
    )


def test_the_cap_survives_a_class_with_one_member() -> None:
    inputs, labels = blobs(per_class=30, classes=2, seed=0)
    labels[0] = 5
    probe = SVMProbe(max_train_samples=20, n_jobs=1)
    probe.fit(
        train_inputs=inputs, train_targets=labels, val_inputs=inputs[:10], val_targets=labels[:10]
    )
    assert probe.pipeline.named_steps['svc'].shape_fit_[0] == 30


def test_default_metrics_use_the_protocol_names() -> None:
    probe = fitted_svm(train=30, val=10)
    inputs, targets = blobs(per_class=10, classes=2, seed=2)
    predictions = probe.predict(inputs=inputs)
    assert probe.default_metrics(predictions=predictions, targets=targets)['accuracy'] == 1.0

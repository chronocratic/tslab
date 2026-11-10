import random

import numpy as np
import pytest
import torch

from tests.ts.unit.probes.helpers import blobs, fitted_svm, ridge_data
from tslab.lab.core.fingerprint import canonicalize
from tslab.ts.probes import RidgeProbe, SVMProbe


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

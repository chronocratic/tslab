"""The SVM classification probe of TS2Vec (Yue et al., AAAI 2022)."""

from dataclasses import dataclass
from typing import override

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import torch

from tslab.interfaces import Probe
from tslab.probes._protocol import (
    cap_samples,
    fit_on_predefined_split,
    PROTOCOL_RANDOM_STATE,
    require_validation,
    to_float64,
    to_numpy,
)
from tslab.probes.metrics import classification_metrics

__all__ = ['SVMProbe', 'SVMProbeSettings']

# Below these sizes the C search is skipped (see SVMProbe).
_MIN_SAMPLES_PER_CLASS = 5
_MIN_SAMPLES = 50
# Near hard margin. libsvm's dual coefficients diverge at C=inf when classes
# overlap; a finite C keeps them bounded.
_HARD_MARGIN_C = 1e6
_C_GRID: list[float] = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, _HARD_MARGIN_C]


@dataclass(frozen=True, kw_only=True)
class SVMProbeSettings:
    """The settings of ``SVMProbe`` that change its results."""

    max_train_samples: int | None


class SVMProbe(Probe):
    """RBF support-vector classifier on frozen representations.

    Features are cast to float64, imputed (constant) and standardised. C is
    chosen from ``{1e-4, ..., 1e4, 1e6}`` on the validation split, then the
    best model is refitted on train and validation together. With fewer than
    5 samples per class or fewer than 50 samples, the search is skipped and
    ``C=1e6`` is fitted on train and validation: a tiny validation split ties
    every C, and the search would keep the smallest one, a constant predictor.
    """

    def __init__(self, *, max_train_samples: int | None = 10_000, n_jobs: int = -1) -> None:
        """Configure the probe.

        Args:
            max_train_samples: Fit on a stratified random subset of at most
                this many training rows; ``None`` uses all of them.
            n_jobs: Parallel jobs of the C search (``-1``: all cores).
        """
        self._settings = SVMProbeSettings(max_train_samples=max_train_samples)
        self._n_jobs = n_jobs
        self._pipeline: Pipeline | None = None

    def fingerprint_settings(self) -> SVMProbeSettings:
        """Return the settings that change results (not ``n_jobs``)."""
        return self._settings

    @property
    def pipeline(self) -> Pipeline:
        """Return the fitted scikit-learn pipeline.

        Raises:
            RuntimeError: Before ``fit``.
        """
        if self._pipeline is None:
            message = 'SVMProbe is not fitted; call fit first'
            raise RuntimeError(message)
        return self._pipeline

    @override
    def fit(
        self,
        *,
        train_inputs: torch.Tensor,
        train_targets: torch.Tensor,
        val_inputs: torch.Tensor | None,
        val_targets: torch.Tensor | None,
    ) -> None:
        """Fit the classifier (see the class docstring)."""
        val_features, val_labels = require_validation(
            val_inputs=val_inputs, val_targets=val_targets, probe='SVMProbe'
        )
        train_features, train_labels = cap_samples(
            features=to_numpy(tensor=train_inputs),
            labels=to_numpy(tensor=train_targets),
            max_samples=self._settings.max_train_samples,
            stratify=True,
            split='train',
        )
        pipeline = Pipeline(
            steps=[
                ('to_float64', to_float64()),
                ('imputer', SimpleImputer(strategy='constant', fill_value=None)),
                ('scaler', StandardScaler()),
                (
                    'svc',
                    SVC(
                        kernel='rbf',
                        coef0=0.0,
                        gamma='scale',
                        shrinking=True,
                        tol=0.001,
                        cache_size=200,
                        class_weight=None,
                        max_iter=10_000_000,
                        decision_function_shape='ovr',
                        random_state=PROTOCOL_RANDOM_STATE,
                        verbose=False,
                    ),
                ),
            ]
        )
        labels = np.concatenate([train_labels, val_labels], axis=0)
        if (
            len(labels) // len(np.unique(labels)) < _MIN_SAMPLES_PER_CLASS
            or len(labels) < _MIN_SAMPLES
        ):
            pipeline.set_params(svc__C=_HARD_MARGIN_C)
            features = np.concatenate([train_features, val_features], axis=0)
            self._pipeline = pipeline.fit(X=features, y=labels)
            return
        self._pipeline = fit_on_predefined_split(
            pipeline=pipeline,
            param_grid={'svc__C': _C_GRID},
            train_features=train_features,
            train_labels=train_labels,
            val_features=val_features,
            val_labels=val_labels,
            scoring=None,
            n_jobs=self._n_jobs,
        )

    @override
    def predict(self, *, inputs: torch.Tensor) -> torch.Tensor:
        """Predict class labels."""
        return torch.from_numpy(self.pipeline.predict(X=to_numpy(tensor=inputs)))

    @override
    def default_metrics(
        self, *, predictions: torch.Tensor, targets: torch.Tensor
    ) -> dict[str, float]:
        """Return accuracy, precision, recall and F1 (see ``classification_metrics``)."""
        return classification_metrics(
            predictions=to_numpy(tensor=predictions), targets=to_numpy(tensor=targets)
        )

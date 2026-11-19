"""The SVM classification probe of TS2Vec."""

from dataclasses import dataclass
import logging
import math
from typing import override

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import torch

from tslab.ml.split import Split
from tslab.ts.probes.metrics import classification_metrics
from tslab.ts.probes.sklearn_probe import (
    cap_samples,
    fit_on_predefined_split,
    PROTOCOL_RANDOM_STATE,
    require_validation,
    SklearnProbe,
    to_float64,
    to_numpy,
)

__all__ = ['SVMProbe', 'SVMProbeSettings']

_LOGGER = logging.getLogger(__name__)
# Below these sizes the C search is skipped (see SVMProbe).
_MIN_SAMPLES_PER_CLASS = 5
_MIN_SAMPLES = 50
_MIN_CLASSES = 2
# Near the hard margin, the dual coefficients of libsvm diverge at C=inf when
# overlap; a finite C keeps them bounded.
_HARD_MARGIN_C = 1e6
_C_GRID: tuple[float, ...] = (0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, _HARD_MARGIN_C)
# In the ValueError that scikit-learn raises when the dual coefficients diverge.
_NOT_FINITE = 'not finite'


@dataclass(frozen=True, kw_only=True)
class SVMProbeSettings:
    """The settings of ``SVMProbe`` that change its results.

    Attributes:
        max_train_samples: The most training rows used in the fit; ``None`` uses all of them.
        fixed_c: The C fitted on every dataset without a search; ``None`` searches C.
    """

    max_train_samples: int | None
    fixed_c: float | None


class SVMProbe(SklearnProbe[SVMProbeSettings]):
    """RBF support-vector classifier on frozen representations.

    Features are cast to float64, imputed (constant) and standardized. C is
    chosen from ``{1e-4, ..., 1e4, 1e6}`` on the validation split, then the
    best model is refitted on train and validation together. With fewer than
    5 samples per class on average or fewer than 50 samples, the search is skipped and
    ``C=inf`` is fitted on train and validation (``C=1e6`` if its coefficients diverge):
    a tiny validation split ties every C, and the search would keep the smallest one, a
    constant predictor. ``fixed_c`` skips the search on every dataset.
    """

    _label = 'SVMProbe'

    def __init__(
        self,
        *,
        max_train_samples: int | None = 10_000,
        fixed_c: float | None = None,
        n_jobs: int = -1,
    ) -> None:
        """Configure the probe.

        Args:
            max_train_samples: Fit on a stratified random subset of at most
                this many training rows; ``None`` uses all of them.
            fixed_c: Fit this C on train and validation without a search, on every
                dataset (``math.inf``: a hard margin); ``None`` searches C.
            n_jobs: Parallel jobs of the C search (``-1``: all cores).
        """
        super().__init__(n_jobs=n_jobs)
        self._settings = SVMProbeSettings(max_train_samples=max_train_samples, fixed_c=fixed_c)

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
            split=Split.TRAIN,
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
        if len(np.unique(labels)) < _MIN_CLASSES:
            message = f'SVMProbe needs at least 2 classes, but the labels hold {np.unique(labels)}'
            raise ValueError(message)
        if self._settings.fixed_c is not None:
            c = self._settings.fixed_c
        elif (
            len(labels) // len(np.unique(labels)) < _MIN_SAMPLES_PER_CLASS
            or len(labels) < _MIN_SAMPLES
        ):
            c = math.inf
        else:
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
            return
        features = np.concatenate([train_features, val_features], axis=0)
        try:
            self._pipeline = pipeline.set_params(svc__C=c).fit(X=features, y=labels)
        except ValueError as error:
            if math.isfinite(c) or _NOT_FINITE not in str(error):
                raise
            _LOGGER.warning('SVMProbe C=inf failed (%s); fitting C=%s', error, _HARD_MARGIN_C)
            self._pipeline = pipeline.set_params(svc__C=_HARD_MARGIN_C).fit(X=features, y=labels)

    @override
    def default_metrics(
        self, *, predictions: torch.Tensor, targets: torch.Tensor
    ) -> dict[str, float]:
        """Return accuracy, precision, recall and F1 (see ``classification_metrics``)."""
        return classification_metrics(
            predictions=to_numpy(tensor=predictions),
            targets=to_numpy(tensor=targets),
            classes=self.pipeline.classes_,
        )

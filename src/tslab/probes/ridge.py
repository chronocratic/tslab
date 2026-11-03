"""The ridge-regression forecasting probe of TS2Vec."""

from dataclasses import dataclass
from typing import override

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch

from tslab.core.enums import Split
from tslab.probes.metrics import forecasting_metrics
from tslab.probes.sklearn_probe import (
    cap_samples,
    fit_on_predefined_split,
    PROTOCOL_RANDOM_STATE,
    require_validation,
    SklearnProbe,
    to_float64,
    to_numpy,
)

__all__ = ['RidgeProbe', 'RidgeProbeSettings']

_ALPHA_GRID: tuple[float, ...] = (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)


def _rmse_plus_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # scikit-learn calls scorer functions positionally as (y_true, y_pred).
    error = y_pred - y_true
    return float(np.sqrt((error**2).mean()) + np.abs(error).mean())


@dataclass(frozen=True, kw_only=True)
class RidgeProbeSettings:
    """The settings of ``RidgeProbe`` that change its results."""

    max_samples: int | None


class RidgeProbe(SklearnProbe[RidgeProbeSettings]):
    """Ridge regression on frozen representations, for forecasting.

    Features are cast to float64, imputed (mean) and standardized. Alpha is
    chosen from ``{0.1, ..., 1000}`` on the validation split by the lowest
    RMSE + MAE, then the best model is refitted on train and validation.
    """

    _label = 'RidgeProbe'

    def __init__(self, *, max_samples: int | None = 100_000, n_jobs: int = -1) -> None:
        """Configure the probe.

        Args:
            max_samples: Fit on a random subset of at most this many rows of
                the training split, and as many of the validation split;
                ``None`` uses every row.
            n_jobs: Parallel jobs of the alpha search (``-1``: all cores).
        """
        super().__init__(n_jobs=n_jobs)
        self._settings = RidgeProbeSettings(max_samples=max_samples)

    @override
    def fit(
        self,
        *,
        train_inputs: torch.Tensor,
        train_targets: torch.Tensor,
        val_inputs: torch.Tensor | None,
        val_targets: torch.Tensor | None,
    ) -> None:
        """Fit the regressor (see the class docstring)."""
        val_features, val_labels = require_validation(
            val_inputs=val_inputs, val_targets=val_targets, probe='RidgeProbe'
        )
        train_features, train_labels = cap_samples(
            features=to_numpy(tensor=train_inputs),
            labels=to_numpy(tensor=train_targets),
            max_samples=self._settings.max_samples,
            stratify=False,
            split=Split.TRAIN,
        )
        val_features, val_labels = cap_samples(
            features=val_features,
            labels=val_labels,
            max_samples=self._settings.max_samples,
            stratify=False,
            split=Split.VAL,
        )
        pipeline = Pipeline(
            steps=[
                ('to_float64', to_float64()),
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('ridge', Ridge(random_state=PROTOCOL_RANDOM_STATE)),
            ]
        )
        self._pipeline = fit_on_predefined_split(
            pipeline=pipeline,
            param_grid={'ridge__alpha': _ALPHA_GRID},
            train_features=train_features,
            train_labels=train_labels,
            val_features=val_features,
            val_labels=val_labels,
            scoring=make_scorer(_rmse_plus_mae, greater_is_better=False),
            n_jobs=self._n_jobs,
        )

    @override
    def default_metrics(
        self, *, predictions: torch.Tensor, targets: torch.Tensor
    ) -> dict[str, float]:
        """Return MSE, MAE and MAPE (see ``forecasting_metrics``)."""
        return forecasting_metrics(
            predictions=to_numpy(tensor=predictions), targets=to_numpy(tensor=targets)
        )

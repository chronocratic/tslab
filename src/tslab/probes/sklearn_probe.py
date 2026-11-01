"""Pieces shared by the scikit-learn probes of the TS2Vec evaluation protocols."""

from collections.abc import Sequence
import logging
from typing import ClassVar, override

import numpy as np
from sklearn.model_selection import GridSearchCV, PredefinedSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import torch

from tslab.interfaces import Probe

__all__ = [
    'PROTOCOL_RANDOM_STATE',
    'SklearnProbe',
    'cap_samples',
    'fit_on_predefined_split',
    'require_validation',
    'to_float64',
    'to_numpy',
]

_LOGGER = logging.getLogger(__name__)
# Every random choice in the protocols uses this seed, so results never depend
# on the global random state.
PROTOCOL_RANDOM_STATE = 42
_MIN_STRATUM = 2


class SklearnProbe[S](Probe):
    """Shared parts of the probes that fit a scikit-learn pipeline.

    Subclasses set ``_label`` and ``_settings``, call ``super().__init__``,
    and implement ``fit`` (storing the result in ``_pipeline``) and
    ``default_metrics``.
    """

    _label: ClassVar[str]
    _settings: S

    def __init__(self, *, n_jobs: int) -> None:
        """Store the worker count of the hyperparameter search."""
        self._n_jobs = n_jobs
        self._pipeline: Pipeline | None = None

    def fingerprint_settings(self) -> S:
        """Return the settings that change results (not ``n_jobs``)."""
        return self._settings

    @property
    def pipeline(self) -> Pipeline:
        """Return the fitted scikit-learn pipeline.

        Raises:
            RuntimeError: Before ``fit``.
        """
        if self._pipeline is None:
            message = f'{self._label} is not fitted; call fit first'
            raise RuntimeError(message)
        return self._pipeline

    @override
    def predict(self, *, inputs: torch.Tensor) -> torch.Tensor:
        """Predict the targets."""
        return torch.from_numpy(self.pipeline.predict(X=to_numpy(tensor=inputs)))


def to_numpy(*, tensor: torch.Tensor) -> np.ndarray:
    """Return ``tensor`` as a numpy array on the CPU."""
    return tensor.detach().cpu().numpy()


def to_float64() -> FunctionTransformer:
    """Return a pipeline step that casts features to float64.

    Near-constant (collapsed) embeddings can have a float64
    ``StandardScaler.scale_`` below the float32 subnormal range. scikit-learn
    casts ``scale_`` to the dtype of the input, so on float32 input it
    underflows to 0 and the division gives inf or NaN. Non-finite inputs are
    left alone on purpose, so a diverged encoder still fails loudly.
    """
    return FunctionTransformer(func=np.asarray, kw_args={'dtype': np.float64})


def cap_samples(
    *, features: np.ndarray, labels: np.ndarray, max_samples: int | None, stratify: bool, split: str
) -> tuple[np.ndarray, np.ndarray]:
    """Keep a random subset of at most ``max_samples`` rows.

    Args:
        features: One row per sample.
        labels: The matching targets.
        max_samples: The cap; ``None`` keeps every row.
        stratify: Keep the class proportions (classification).
        split: Split name for the log message.

    Returns:
        The kept features and labels.
    """
    if max_samples is None or len(features) <= max_samples:
        return features, labels
    _LOGGER.info(
        'the %s split has %d samples; evaluating on a random subset of %d',
        split,
        len(features),
        max_samples,
    )
    # A class with a single member cannot be split proportionally.
    if stratify:
        stratify = bool(np.unique(labels, return_counts=True)[1].min() >= _MIN_STRATUM)
    kept_features, _, kept_labels, _ = train_test_split(
        features,
        labels,
        train_size=max_samples,
        random_state=PROTOCOL_RANDOM_STATE,
        stratify=labels if stratify else None,
    )
    return kept_features, kept_labels


def require_validation(
    *, val_inputs: torch.Tensor | None, val_targets: torch.Tensor | None, probe: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return the validation split as numpy arrays.

    Raises:
        ValueError: If either part is missing or the split has no rows.
    """
    if val_inputs is None or val_targets is None:
        message = (
            f'{probe} chooses its hyperparameter on a validation split; the representations '
            'have no val_representations and val_targets'
        )
        raise ValueError(message)
    if len(val_inputs) == 0:
        message = f'{probe} needs at least one validation row, but the validation split is empty'
        raise ValueError(message)
    return to_numpy(tensor=val_inputs), to_numpy(tensor=val_targets)


def fit_on_predefined_split(  # noqa: PLR0913 # why: each half of the split is a separate argument
    *,
    pipeline: Pipeline,
    param_grid: dict[str, Sequence[float]],
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    scoring: object | None,
    n_jobs: int,
) -> Pipeline:
    """Grid-search ``pipeline`` on the train/validation split, then refit on both.

    Each candidate is fitted on the training rows and scored on the validation
    rows (a scikit-learn ``PredefinedSplit``); the best one is refitted on all
    rows.

    Returns:
        The refitted best pipeline.
    """
    features = np.concatenate([train_features, val_features], axis=0)
    labels = np.concatenate([train_labels, val_labels], axis=0)
    split = PredefinedSplit(test_fold=[-1] * len(train_features) + [0] * len(val_features))
    search = GridSearchCV(
        estimator=pipeline, param_grid=param_grid, scoring=scoring, cv=split, n_jobs=n_jobs
    )
    search.fit(X=features, y=labels)
    best = search.best_estimator_
    if not isinstance(best, Pipeline):  # a clone of the pipeline passed in
        message = f'expected the best estimator to be a Pipeline, got {type(best).__name__}'
        raise TypeError(message)
    return best

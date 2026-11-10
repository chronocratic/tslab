"""Shared helpers of the probe tests."""

import torch

from tslab.ts.probes import SVMProbe

__all__ = ['blobs', 'fitted_svm', 'ridge_data']


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

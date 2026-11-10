"""The ready ``evaluate`` stage."""

from collections.abc import Mapping, Sequence

import torch

from tslab.core.artifact import Metrics, Tensors
from tslab.core.paths import InstanceName, normalize_instance_name
from tslab.core.run import Run
from tslab.core.stage import stage
from tslab.interfaces import DownstreamModel, Metric

__all__ = ['evaluate']

STAGE_NAME = 'evaluate'


def _tensor(*, tensors: Mapping[str, torch.Tensor], key: str) -> torch.Tensor:
    if key not in tensors:
        message = f'the representations have no {key!r}; available: {sorted(tensors)}'
        raise KeyError(message)
    return tensors[key]


def _scores(
    *, metrics: Sequence[Metric], predictions: torch.Tensor, targets: torch.Tensor
) -> dict[str, float]:
    values: dict[str, float] = {}
    for metric in metrics:
        if metric.name in values:
            message = f'two metrics are named {metric.name!r}'
            raise ValueError(message)
        values[metric.name] = float(metric(predictions=predictions, targets=targets))
    return values


@stage(name=STAGE_NAME, output=Metrics)
def evaluate(
    *,
    run: Run,
    name: InstanceName | None = None,
    representations: Tensors,
    downstream_model: DownstreamModel,
    metrics: Sequence[Metric] | None = None,
) -> Metrics:
    """Fit ``downstream_model`` on the representations and score it on the test split.

    Args:
        run: The current run; its trackers receive the metrics.
        name: Instance name, for example ``('forecasting', 'h24')``.
        representations: From ``encode`` or the forecasting stages:
            ``train_*`` and ``test_*`` keys, and ``val_*`` when there is a
            validation split.
        downstream_model: For example ``tslab.probes.SVMProbe()``.
        metrics: Scores to compute; ``None`` uses the protocol metrics of
            ``downstream_model``.

    Returns:
        The metric values.
    """
    tensors = representations.tensors
    downstream_model.fit(
        train_inputs=_tensor(tensors=tensors, key='train_representations'),
        train_targets=_tensor(tensors=tensors, key='train_targets'),
        val_inputs=tensors.get('val_representations'),
        val_targets=tensors.get('val_targets'),
    )
    targets = _tensor(tensors=tensors, key='test_targets')
    predictions = downstream_model.predict(
        inputs=_tensor(tensors=tensors, key='test_representations')
    )
    values = (
        downstream_model.default_metrics(predictions=predictions, targets=targets)
        if metrics is None
        else _scores(metrics=metrics, predictions=predictions, targets=targets)
    )
    prefix = '/'.join((STAGE_NAME, *normalize_instance_name(name=name)))
    run.log_metrics(metrics={f'{prefix}/{key}': value for key, value in values.items()})
    return Metrics(data=values)

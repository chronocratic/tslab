"""Probe: score a checkpoint trained by another run with an SVM probe.

This is the second step of the ``chain`` study: ``classify`` trains a model, then
this script loads the checkpoint it saved, encodes every split and scores the
probe. Run ``python -m {{package_name}}.probe --help`` for every setting.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import auto, StrEnum
from pathlib import Path

import torch
import tslab
from tslab.adapters.chronocratic.datasets import (
    build_chronocratic_data_source,
    ChronocraticClassificationData,
    ChronocraticDataChoice,
)
from tslab.adapters.chronocratic.models import build_chronocratic_model, ChronocraticEncoder
from tslab.interfaces import Metric, Probe
from tslab.lightning import LightningTrainer, LightningTrainerConfig
from tslab.probes import RidgeProbe, SVMProbe
from tslab.tracking import build_trackers, TrackingOptions

from {{package_name}}.models import Models


class ProbeHead(StrEnum):
    """The supported classification probes in the chain study."""

    SVM = auto()
    RIDGE = auto()


@dataclass(frozen=True, kw_only=True)
class ProbeConfig:
    """One probe run. Every field is a command-line flag.

    Attributes:
        model: The model the checkpoint was trained with.
        data: A classification dataset (``data:ucr`` or ``data:uea``).
        checkpoint: The checkpoint folder of the training run.
        head: The classification probe to fit.
        max_train_samples: Fit the SVM on at most this many training rows.
        trainer: Lightning settings; only used to run the encoder.
        encoder: How representations are computed.
        tracking: Experiment trackers besides the always-on CSV files.
        run: Resume and rerun options.
        run_path: Run folder under ``output_root``.
        output_root: Folder holding all runs.
        seed: Seeds Python, NumPy and torch at the start of the run.
    """

    model: Models
    data: ChronocraticDataChoice
    checkpoint: Path
    head: ProbeHead
    max_train_samples: int = 10_000
    trainer: LightningTrainerConfig = field(default_factory=LightningTrainerConfig)
    encoder: ChronocraticEncoder = field(default_factory=ChronocraticEncoder)
    tracking: TrackingOptions = field(default_factory=TrackingOptions)
    run: tslab.RunOptions = field(default_factory=tslab.RunOptions)
    run_path: Path = Path('probe')
    output_root: Path = Path('runs')
    seed: int = 0


@dataclass(frozen=True, kw_only=True)
class RidgeClassificationAccuracy:
    """Round ridge predictions to the nearest observed class before scoring."""

    classes: tuple[float, ...]

    @property
    def name(self) -> str:
        """Return the metric name used in evaluation output."""
        return 'ridge_accuracy'

    def __call__(self, *, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Score predictions after rounding each one to the nearest class.

        Args:
            predictions: Ridge regression predictions.
            targets: Expected class labels.

        Returns:
            Classification accuracy as a float.
        """
        classes = torch.tensor(self.classes, dtype=predictions.dtype, device=predictions.device)
        nearest = classes[(predictions.reshape(-1, 1) - classes).abs().argmin(dim=1)]
        return float((nearest == targets.reshape(-1)).float().mean().item())


def _svm_probe(*, max_train_samples: int) -> Probe:
    return SVMProbe(max_train_samples=max_train_samples)


def _ridge_probe(*, max_train_samples: int) -> Probe:
    return RidgeProbe(max_samples=max_train_samples)


def _svm_metrics(*, representations: tslab.Tensors) -> Sequence[Metric] | None:
    del representations
    return None


def _ridge_metrics(*, representations: tslab.Tensors) -> Sequence[Metric]:
    targets = representations.tensors['train_targets']
    classes = tuple(float(value) for value in torch.unique(targets).tolist())
    return (RidgeClassificationAccuracy(classes=classes),)


_PROBES: Mapping[ProbeHead, Callable[..., Probe]] = {
    ProbeHead.SVM: _svm_probe,
    ProbeHead.RIDGE: _ridge_probe,
}
_METRICS: Mapping[ProbeHead, Callable[..., Sequence[Metric] | None]] = {
    ProbeHead.SVM: _svm_metrics,
    ProbeHead.RIDGE: _ridge_metrics,
}


def main(*, config: ProbeConfig) -> None:
    """Load the checkpoint, encode every split and score the SVM probe.

    Raises:
        TypeError: If ``config.data`` is not a classification dataset.
    """
    trackers = build_trackers(options=config.tracking)
    with tslab.Run(
        config=config, root=config.output_root, relative_path=config.run_path, trackers=trackers
    ) as run:
        data = build_chronocratic_data_source(config=config.data)
        if not isinstance(data, ChronocraticClassificationData):
            message = 'probe needs a classification dataset: data:ucr or data:uea'
            raise TypeError(message)
        model = build_chronocratic_model(config=config.model, data_info=data.info)
        trainer = LightningTrainer(config=config.trainer)
        checkpoint = tslab.Checkpoint.external(path=config.checkpoint, trusted=True)
        representations = tslab.stages.encode(
            run=run,
            trainer=trainer,
            model=model,
            checkpoint=checkpoint,
            data=data,
            encoder=config.encoder,
        )
        downstream_model = _PROBES[config.head](max_train_samples=config.max_train_samples)
        metrics = _METRICS[config.head](representations=representations)
        tslab.stages.evaluate(
            run=run,
            representations=representations,
            downstream_model=downstream_model,
            metrics=metrics,
        )


if __name__ == '__main__':
    main(config=tslab.parse_config(config_class=ProbeConfig))

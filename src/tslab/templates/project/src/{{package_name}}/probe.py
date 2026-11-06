"""Probe: score a checkpoint trained by another run with an SVM probe.

This is the second step of the ``chain`` study: ``classify`` trains a model, then
this script loads the checkpoint it saved, encodes every split and scores the
probe. Run ``python -m {{package_name}}.probe --help`` for every setting.
"""

from dataclasses import dataclass, field
from pathlib import Path

import tslab
from tslab.adapters.chronocratic.datasets import (
    build_chronocratic_data_source,
    ChronocraticClassificationData,
    ChronocraticDataChoice,
)
from tslab.adapters.chronocratic.models import build_chronocratic_model, ChronocraticEncoder
from tslab.lightning import LightningTrainer, LightningTrainerConfig
from tslab.probes import SVMProbe
from tslab.tracking import build_trackers, TrackingOptions

from {{package_name}}.models import Models


@dataclass(frozen=True, kw_only=True)
class ProbeConfig:
    """One probe run. Every field is a command-line flag.

    Attributes:
        model: The model the checkpoint was trained with.
        data: A classification dataset (``data:ucr`` or ``data:uea``).
        checkpoint: The checkpoint folder of the training run.
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
    max_train_samples: int = 10_000
    trainer: LightningTrainerConfig = field(default_factory=LightningTrainerConfig)
    encoder: ChronocraticEncoder = field(default_factory=ChronocraticEncoder)
    tracking: TrackingOptions = field(default_factory=TrackingOptions)
    run: tslab.RunOptions = field(default_factory=tslab.RunOptions)
    run_path: Path = Path('probe')
    output_root: Path = Path('runs')
    seed: int = 0


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
        tslab.stages.evaluate(
            run=run,
            representations=representations,
            downstream_model=SVMProbe(max_train_samples=config.max_train_samples),
        )


if __name__ == '__main__':
    main(config=tslab.parse_config(config_class=ProbeConfig))

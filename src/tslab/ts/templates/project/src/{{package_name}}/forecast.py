"""Forecasting: train a model, encode the series, score a ridge probe per horizon.

Run ``python -m {{package_name}}.forecast --help`` for the presets, and
``python -m {{package_name}}.forecast no-preset --help`` for every setting.

Models whose encoder can encode a whole series step by step (TS2Vec, AutoTCL,
CoST) encode each series once for all horizons. The others encode windows of
``window_length`` steps, once per horizon.
"""

from dataclasses import dataclass, field
from pathlib import Path

import tslab.lab
import tslab.ml
from tslab.ml.lightning import LightningTrainer, LightningTrainerConfig
import tslab.ml.stages
from tslab.ml.tracking import build_trackers, TrackingOptions
import tslab.ts
from tslab.ts.adapters.chronocratic.datasets import (
    build_chronocratic_data_source,
    ChronocraticDataChoice,
    ChronocraticForecastingData,
    data_settings,
)
from tslab.ts.adapters.chronocratic.models import (
    build_chronocratic_model,
    ChronocraticEncoder,
    model_settings,
)
from tslab.ts.probes import RidgeProbe
import tslab.ts.stages
from tslab.ts.tasks import forecasting_representations

from {{package_name}}.models import Models


@dataclass(frozen=True, kw_only=True)
class ForecastConfig:
    """One forecasting run. Every field is a command-line flag.

    Attributes:
        model: The model and its settings (``model:ts2vec``, ``model:cost``, ...).
        data: A forecasting dataset (``data:ett``, ``data:electricity``, ``data:weather``).
        horizons: Steps ahead to forecast; each gets its own probe and metrics.
        window_length: Input window for models that encode windows, not whole series.
        trainer: Lightning training settings.
        encoder: How representations are computed.
        tracking: Experiment trackers besides the always-on CSV files.
        run: Resume and rerun options.
        run_path: Run folder under ``output_root``, for example ``ts2vec/ETTh1/seed_0``.
        output_root: Folder holding all runs.
        seed: Seeds Python, NumPy and torch at the start of the run.
    """

    model: Models
    data: ChronocraticDataChoice
    horizons: tuple[int, ...] = (24, 48, 168, 336, 720)
    window_length: int = 96
    trainer: LightningTrainerConfig = field(default_factory=LightningTrainerConfig)
    encoder: ChronocraticEncoder = field(default_factory=ChronocraticEncoder)
    tracking: TrackingOptions = field(default_factory=TrackingOptions)
    run: tslab.lab.RunOptions = field(default_factory=tslab.lab.RunOptions)
    run_path: Path = Path('forecasting')
    output_root: Path = Path('runs')
    seed: int = 0


TOY_HOURLY = Path('data') / 'toy_hourly.csv.gz'

PRESETS = {
    'ts2vec-full-series': (
        'TS2Vec on the toy hourly series, trained on the whole training series',
        ForecastConfig(
            model=model_settings(choice=Models, name='ts2vec'),
            data=data_settings(name='ett', dataset_file_path=TOY_HOURLY, variant='ETTh1'),
            run_path=Path('ts2vec') / 'toy_hourly' / 'full-series',
        ),
    ),
    'cost-cropped': (
        'CoST on the toy hourly series, trained on random 128-step crops, batch size 1',
        ForecastConfig(
            model=model_settings(
                choice=Models, name='cost', max_train_length=128, sequence_length=128
            ),
            data=data_settings(
                name='ett', dataset_file_path=TOY_HOURLY, variant='ETTh1', batch_size=1
            ),
            run_path=Path('cost') / 'toy_hourly' / 'cropped',
        ),
    ),
}


def main(*, config: ForecastConfig) -> None:
    """Train, encode and evaluate every horizon.

    Raises:
        TypeError: If ``config.data`` is not a forecasting dataset.
    """
    trackers = build_trackers(options=config.tracking)
    with tslab.lab.Run(
        config=config, root=config.output_root, relative_path=config.run_path, trackers=trackers
    ) as run:
        data = build_chronocratic_data_source(config=config.data)
        if not isinstance(data, ChronocraticForecastingData):
            message = (
                'forecast needs a forecasting dataset: data:ett, data:electricity or data:weather'
            )
            raise TypeError(message)
        model = build_chronocratic_model(config=config.model, data_info=data.info)
        trainer = LightningTrainer(config=config.trainer)
        checkpoint = tslab.ml.stages.train(run=run, trainer=trainer, model=model, data=data)
        by_horizon = forecasting_representations(
            run=run,
            trainer=trainer,
            model=model,
            checkpoint=checkpoint,
            data=data,
            encoder=config.encoder,
            horizons=config.horizons,
            window_length=config.window_length,
        )
        for horizon, representations in by_horizon.items():
            tslab.ts.stages.evaluate(
                run=run,
                name=f'h{horizon}',
                representations=representations,
                downstream_model=RidgeProbe(),
            )


if __name__ == '__main__':
    main(config=tslab.lab.parse_config(config_class=ForecastConfig, presets=PRESETS))

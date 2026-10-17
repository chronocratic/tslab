"""Input smoothing for forecasting representations, with your own transform stage.

TS2Vec is trained on a synthetic hourly series. A transform stage then smooths
the target channels of the whole series with a trailing moving average, the
smoothed series is encoded, and a ridge probe is fitted and scored on it, per
horizon. Targets always come from the raw series.

What it shows:

- **Your own stage with several instances** (``transform``, one per
  smoothing). One smoothing is broken on purpose (``smooth-broken`` has a
  negative window): it fails, everything that depends on it is skipped, and
  the other smoothings finish. Fix it and run again: only the fixed smoothing
  and its dependants run.
- **A changed series as a stage input.** The transform returns a whole series;
  ``forecasting_representations(..., series=...)`` encodes it instead of the
  series of the data itself, so the encoding depends on the transform like on any input.
- **Frozen channels.** Covariate channels (time features) are known in advance,
  so the transform only changes ``data.info.target_channels``.
- **Nested instance names**: ``evaluate[forecasting/smooth-3/h24]`` is
  task / smoothing / horizon.
- **A preprocessing step as its own stage.** The transform is separate from
  encoding and evaluation, so changing how its result is evaluated never
  recomputes it.

Run it from the root of a TSLab checkout (it reads the toy series in
``tests/ts/adapters/data``); results go to ``runs/advanced``::

    uv run python examples/advanced/main.py
    uv run python examples/advanced/main.py --fix-broken-smoothing
    uv run tslab status runs/advanced
"""

from dataclasses import dataclass, field, replace
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812 # why: the usual torch alias

import tslab.lab
import tslab.ml
from tslab.ml.lightning import LightningTrainer, LightningTrainerConfig
import tslab.ml.stages
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
    ChronocraticModelChoice,
    model_settings,
)
from tslab.ts.probes import RidgeProbe
import tslab.ts.stages
from tslab.ts.tasks import forecasting_representations

TOY_HOURLY = Path(__file__).parents[2] / 'tests' / 'ts' / 'adapters' / 'data' / 'toy_hourly.csv.gz'
TASK = 'forecasting'


@dataclass(frozen=True, kw_only=True)
class Smoothing:
    """One trailing moving average.

    Attributes:
        name: Instance name of the smoothing.
        window: Steps averaged, the current one included; must be positive.
    """

    name: str
    window: int


@dataclass(frozen=True, kw_only=True)
class AdvancedConfig:
    """Settings of the example; every field is a command-line flag.

    Attributes:
        model: The model; TS2Vec on 512-step training crops by default.
        data: The forecasting dataset.
        smoothings: The smoothings; one is broken unless ``fix_broken_smoothing`` is set.
        fix_broken_smoothing: Give the broken smoothing a valid window.
        horizons: Steps ahead to forecast.
        trainer: Lightning training settings.
        encoder: How representations are computed.
        root: Folder holding all runs.
        relative_path: The folder of this run under ``root``.
        seed: Seeds Python, NumPy and torch.
        run: Resume and rerun options.
    """

    model: ChronocraticModelChoice = field(
        default_factory=lambda: model_settings(
            choice=ChronocraticModelChoice, name='ts2vec', max_train_length=512
        )
    )
    data: ChronocraticDataChoice = field(
        default_factory=lambda: data_settings(
            name='ett', dataset_file_path=TOY_HOURLY, variant='ETTh1'
        )
    )
    smoothings: tuple[Smoothing, ...] = (
        Smoothing(name='smooth-3', window=3),
        Smoothing(name='smooth-9', window=9),
        Smoothing(name='smooth-broken', window=-5),
    )
    fix_broken_smoothing: bool = False
    horizons: tuple[int, ...] = (24, 48)
    trainer: LightningTrainerConfig = field(
        default_factory=lambda: LightningTrainerConfig(max_epochs=5, min_epochs=0)
    )
    encoder: ChronocraticEncoder = field(default_factory=ChronocraticEncoder)
    root: Path = Path('runs')
    relative_path: Path = Path('advanced')
    seed: int = 0
    run: tslab.lab.RunOptions = field(default_factory=tslab.lab.RunOptions)


@tslab.lab.stage(name='transform', output=tslab.ml.Tensors)
def transform(
    *,
    run: tslab.lab.Run,
    name: tslab.lab.InstanceName | None = None,
    data: ChronocraticForecastingData,
    smoothing: Smoothing,
) -> tslab.ml.Tensors:
    """Smooth the target channels of the whole series with a trailing moving average.

    Trailing means each step averages only itself and the steps before it, so
    no future value leaks into the input. Covariate channels stay as they are.

    Returns:
        ``series``: the whole series, with smoothed target channels.

    Raises:
        ValueError: If ``smoothing.window`` is not positive.
    """
    del run, name
    if smoothing.window <= 0:
        message = f'{smoothing.name}: window must be positive, got {smoothing.window}'
        raise ValueError(message)
    series = data.full_series().to(dtype=torch.float32)
    targets = list(data.info.target_channels)
    # (N, T, C) to (N, C, T); repeat the first step so the average is defined from the start
    channels = series[:, :, targets].transpose(1, 2)
    padded = F.pad(channels, (smoothing.window - 1, 0), mode='replicate')
    smoothed = series.clone()
    averaged = F.avg_pool1d(padded, kernel_size=smoothing.window, stride=1)
    smoothed[:, :, targets] = averaged.transpose(1, 2)
    return tslab.ml.Tensors(tensors={'series': smoothed})


def main(*, config: AdvancedConfig) -> dict[str, float]:
    """Train, smooth, encode and evaluate; return the test MSE per ``<variant>/h<horizon>``.

    Raises:
        TypeError: If ``config.data`` is not a forecasting dataset.
    """
    smoothings = tuple(
        replace(item, window=abs(item.window)) if config.fix_broken_smoothing else item
        for item in config.smoothings
    )
    errors: dict[str, float] = {}
    with tslab.lab.Run(config=config, root=config.root, relative_path=config.relative_path) as run:
        data = build_chronocratic_data_source(config=config.data)
        if not isinstance(data, ChronocraticForecastingData):
            message = 'this example needs a forecasting dataset'
            raise TypeError(message)
        model = build_chronocratic_model(config=config.model, data_info=data.info)
        trainer = LightningTrainer(config=config.trainer)
        checkpoint = tslab.ml.stages.train(run=run, trainer=trainer, model=model, data=data)
        clean = forecasting_representations(
            run=run,
            name=(TASK, 'clean'),
            trainer=trainer,
            model=model,
            checkpoint=checkpoint,
            data=data,
            encoder=config.encoder,
            horizons=config.horizons,
        )
        variants = {'clean': clean}
        for smoothing in smoothings:
            smoothed = transform(
                run=run, name=(TASK, smoothing.name), data=data, smoothing=smoothing
            )
            variants[smoothing.name] = forecasting_representations(
                run=run,
                name=(TASK, smoothing.name),
                trainer=trainer,
                model=model,
                checkpoint=checkpoint,
                data=data,
                encoder=config.encoder,
                horizons=config.horizons,
                series=smoothed,
            )
        for variant, per_horizon in variants.items():
            for horizon, representations in per_horizon.items():
                metrics = tslab.ts.stages.evaluate(
                    run=run,
                    name=(TASK, variant, f'h{horizon}'),
                    representations=representations,
                    downstream_model=RidgeProbe(),
                )
                if tslab.lab.is_available(artifact=metrics):
                    errors[f'{variant}/h{horizon}'] = metrics.values['mse_loss']
    return errors


if __name__ == '__main__':
    for key, error in main(config=tslab.lab.parse_config(config_class=AdvancedConfig)).items():
        print(f'{key}: mse {error:.4f}')

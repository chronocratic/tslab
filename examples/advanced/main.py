"""Adversarial robustness of forecasting representations, with your own attack stage.

TS2Vec is trained on a synthetic hourly series. A one-step FGSM attack then
perturbs the test part of the series so that its representation moves as far
as possible. A ridge probe is fitted on clean training representations and
scored on the clean and on each attacked test split, per horizon.

What it shows:

- **Your own stage with several instances** (``attack``, one per attack). One
  attack is broken on purpose (``fgsm-broken`` has a negative step): it fails,
  everything that depends on it is skipped, and the other attacks finish.
  Fix it and run again: only the fixed attack and its dependants run.
- **A changed series as a stage input.** The attack returns a whole series;
  ``forecasting_representations(..., series=...)`` encodes it instead of the
  series of the data itself, so the encoding depends on the attack like on any input.
- **Frozen channels.** Covariate channels (time features) are known in advance,
  so the attack only changes ``data.info.target_channels``.
- **Nested instance names**: ``evaluate[forecasting/fgsm-small/h24]`` is
  task / attack / horizon.
- **An expensive step as its own stage.** The attack (gradients through the
  encoder) is separate from encoding and evaluation, so changing how its
  result is evaluated never recomputes it.

Run it from the root of a TSLab checkout (it reads the toy series in
``tests/ts/adapters/data``); results go to ``runs/advanced``::

    uv run python examples/advanced/main.py
    uv run python examples/advanced/main.py --fix-broken-attack
    uv run tslab status runs/advanced
"""

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol, runtime_checkable

from chronocratic.models import EncodingOutputShape
import torch

import tslab.lab
import tslab.ml
from tslab.ml.lightning import LightningTrainer, LightningTrainerConfig
from tslab.ml.split import Split
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
    FingerprintedModel,
    model_settings,
)
from tslab.ts.probes import RidgeProbe
import tslab.ts.stages
from tslab.ts.tasks import forecasting_representations

TOY_HOURLY = Path(__file__).parents[2] / 'tests' / 'ts' / 'adapters' / 'data' / 'toy_hourly.csv.gz'
TASK = 'forecasting'


@dataclass(frozen=True, kw_only=True)
class FgsmAttack:
    """One FGSM attack.

    Attributes:
        name: Instance name of the attack.
        epsilon: Step size in units of the scaled data; must be positive.
    """

    name: str
    epsilon: float


@dataclass(frozen=True, kw_only=True)
class AdvancedConfig:
    """Settings of the example; every field is a command-line flag.

    Attributes:
        model: The model; TS2Vec on 512-step training crops by default.
        data: The forecasting dataset.
        attacks: The attacks; one is broken unless ``fix_broken_attack`` is set.
        fix_broken_attack: Give the broken attack a valid step.
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
    attacks: tuple[FgsmAttack, ...] = (
        FgsmAttack(name='fgsm-small', epsilon=0.01),
        FgsmAttack(name='fgsm-large', epsilon=0.1),
        FgsmAttack(name='fgsm-broken', epsilon=-0.05),
    )
    fix_broken_attack: bool = False
    horizons: tuple[int, ...] = (24, 48)
    trainer: LightningTrainerConfig = field(
        default_factory=lambda: LightningTrainerConfig(max_epochs=5, min_epochs=0)
    )
    encoder: ChronocraticEncoder = field(default_factory=ChronocraticEncoder)
    root: Path = Path('runs')
    relative_path: Path = Path('advanced')
    seed: int = 0
    run: tslab.lab.RunOptions = field(default_factory=tslab.lab.RunOptions)


@runtime_checkable
class BatchEncoding(Protocol):
    """The differentiable encoding of one batch by chronocratic."""

    def encode_batch(self, batch_x: torch.Tensor, *, output: EncodingOutputShape) -> torch.Tensor:
        """Encode ``batch_x`` with gradients."""
        ...


@tslab.lab.stage(name='attack', output=tslab.ml.Tensors)
def attack(  # noqa: PLR0913 # why: a stage takes its inputs as keyword arguments
    *,
    run: tslab.lab.Run,
    name: tslab.lab.InstanceName | None = None,
    trainer: LightningTrainer,
    model: FingerprintedModel,
    checkpoint: tslab.ml.Checkpoint,
    data: ChronocraticForecastingData,
    fgsm: FgsmAttack,
) -> tslab.ml.Tensors:
    """Perturb the test part of the series to move its representation away (one FGSM step).

    The loss is the squared distance between the representations of the
    perturbed and the clean test series, so no labels or probe are needed.
    Only target channels move; covariate channels stay as they are.

    Returns:
        ``series``: the whole series, with the test part attacked.

    Raises:
        ValueError: If ``fgsm.epsilon`` is not positive.
        TypeError: If the model has no differentiable ``encode_batch``.
    """
    del name
    if fgsm.epsilon <= 0:
        message = f'{fgsm.name}: epsilon must be positive, got {fgsm.epsilon}'
        raise ValueError(message)
    trainer.load(model=model, checkpoint=checkpoint)
    model.eval()
    if not isinstance(model, BatchEncoding):
        message = f'{type(model).__qualname__} has no encode_batch'
        raise TypeError(message)
    series = data.full_series().to(dtype=torch.float32)
    test = data.split_slices()[Split.TEST]
    clean = series[:, test]
    movable = torch.zeros(series.shape[-1])
    movable[list(data.info.target_channels)] = 1.0
    with torch.no_grad():
        reference = model.encode_batch(clean, output=EncodingOutputShape.VECTOR)
    # The drift loss has zero gradient at the clean point, so start a tiny step away from it.
    generator = torch.Generator().manual_seed(run.seed)
    start = clean + 1e-3 * torch.randn(clean.shape, generator=generator) * movable
    start.requires_grad_()
    drift = (model.encode_batch(start, output=EncodingOutputShape.VECTOR) - reference).pow(2).sum()
    (gradient,) = torch.autograd.grad(outputs=drift, inputs=start)
    attacked = series.clone()
    attacked[:, test] = clean + fgsm.epsilon * gradient.sign() * movable
    return tslab.ml.Tensors(tensors={'series': attacked})


@tslab.lab.stage(name='attacked_test_split', output=tslab.ml.Tensors)
def attacked_test_split(
    *,
    run: tslab.lab.Run,
    name: tslab.lab.InstanceName | None = None,
    clean: tslab.ml.Tensors,
    attacked: tslab.ml.Tensors,
) -> tslab.ml.Tensors:
    """Keep the clean training and validation splits and take the attacked test split.

    The probe is then fitted on clean representations and scored on attacked ones.
    """
    del run, name
    test = {key: value for key, value in attacked.tensors.items() if key.startswith('test_')}
    return tslab.ml.Tensors(tensors={**clean.tensors, **test})


def main(*, config: AdvancedConfig) -> dict[str, float]:
    """Train, attack, encode and evaluate; return the test MSE per ``<variant>/h<horizon>``.

    Raises:
        TypeError: If ``config.data`` is not a forecasting dataset.
    """
    attacks = tuple(
        replace(fgsm, epsilon=abs(fgsm.epsilon)) if config.fix_broken_attack else fgsm
        for fgsm in config.attacks
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
        for fgsm in attacks:
            attacked_series = attack(
                run=run,
                name=(TASK, fgsm.name),
                trainer=trainer,
                model=model,
                checkpoint=checkpoint,
                data=data,
                fgsm=fgsm,
            )
            attacked = forecasting_representations(
                run=run,
                name=(TASK, fgsm.name),
                trainer=trainer,
                model=model,
                checkpoint=checkpoint,
                data=data,
                encoder=config.encoder,
                horizons=config.horizons,
                series=attacked_series,
            )
            variants[fgsm.name] = {
                horizon: attacked_test_split(
                    run=run,
                    name=(TASK, fgsm.name, f'h{horizon}'),
                    clean=clean[horizon],
                    attacked=attacked[horizon],
                )
                for horizon in config.horizons
            }
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

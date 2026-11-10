"""Classification: train a model, encode every split, score an SVM probe.

Run ``python -m {{package_name}}.classify --help`` for the presets, and
``python -m {{package_name}}.classify no-preset --help`` for every setting.
The first word is either ``no-preset`` (choose model and dataset yourself) or a
preset from ``PRESETS``, which fixes its model and dataset; flags after it
change any value.
"""

from dataclasses import dataclass, field
from pathlib import Path

from chronocratic.datasets import ClassificationSplitMode
import tslab.lab
import tslab.ml
from tslab.ml.lightning import LightningTrainer, LightningTrainerConfig
from tslab.ml.tracking import build_trackers, TrackingOptions
import tslab.ts
from tslab.ts.adapters.chronocratic.datasets import (
    build_chronocratic_data_source,
    ChronocraticClassificationData,
    ChronocraticDataChoice,
    data_settings,
)
from tslab.ts.adapters.chronocratic.models import (
    build_chronocratic_model,
    ChronocraticEncoder,
    model_settings,
)
from tslab.ts.probes import SVMProbe

from {{package_name}}.models import Models
from {{package_name}}.stages import add_noise


@dataclass(frozen=True, kw_only=True)
class ClassifyConfig:
    """One classification run. Every field is a command-line flag.

    Attributes:
        model: The model and its settings (``model:ts2vec``, ``model:tstcc``, ...).
        data: A classification dataset (``data:ucr`` or ``data:uea``).
        trainer: Lightning training settings.
        encoder: How representations are computed.
        noise_levels: Standard deviations for the noisy-test robustness check.
        tracking: Experiment trackers besides the always-on CSV files.
        run: Resume and rerun options.
        run_path: Run folder under ``output_root``; use one folder per model,
            dataset and seed, for example ``ts2vec/Coffee/seed_0``.
        output_root: Folder holding all runs.
        seed: Seeds Python, NumPy and torch at the start of the run.
    """

    model: Models
    data: ChronocraticDataChoice
    trainer: LightningTrainerConfig = field(default_factory=LightningTrainerConfig)
    encoder: ChronocraticEncoder = field(default_factory=ChronocraticEncoder)
    noise_levels: tuple[float, ...] = (0.1, 0.5)
    tracking: TrackingOptions = field(default_factory=TrackingOptions)
    run: tslab.lab.RunOptions = field(default_factory=tslab.lab.RunOptions)
    run_path: Path = Path('classification')
    output_root: Path = Path('runs')
    seed: int = 0


TOY_SPECTRA = Path('data') / 'ToySpectra'

PRESETS = {
    'ts2vec-archive-split': (
        'TS2Vec on the toy spectra, the train/test split of the archive',
        ClassifyConfig(
            model=model_settings(choice=Models, name='ts2vec'),
            data=data_settings(
                name='ucr', dataset_folder_path=TOY_SPECTRA, target_column_name='target'
            ),
            run_path=Path('ts2vec') / 'ToySpectra' / 'archive-split',
        ),
    ),
    'ts2vec-stratified-resplit': (
        'TS2Vec on the toy spectra, train and test merged and re-split 50/50, batch size 16',
        ClassifyConfig(
            model=model_settings(choice=Models, name='ts2vec'),
            data=data_settings(
                name='ucr',
                dataset_folder_path=TOY_SPECTRA,
                target_column_name='target',
                splitting_strategy=ClassificationSplitMode.MANUAL,
                batch_size=16,
            ),
            run_path=Path('ts2vec') / 'ToySpectra' / 'stratified-resplit',
        ),
    ),
}


def main(*, config: ClassifyConfig) -> None:
    """Train, encode and evaluate; then evaluate again on noisy test representations.

    Raises:
        TypeError: If ``config.data`` is not a classification dataset.
    """
    trackers = build_trackers(options=config.tracking)
    with tslab.lab.Run(
        config=config, root=config.output_root, relative_path=config.run_path, trackers=trackers
    ) as run:
        data = build_chronocratic_data_source(config=config.data)
        if not isinstance(data, ChronocraticClassificationData):
            message = 'classify needs a classification dataset: data:ucr or data:uea'
            raise TypeError(message)
        model = build_chronocratic_model(config=config.model, data_info=data.info)
        trainer = LightningTrainer(config=config.trainer)
        checkpoint = tslab.ml.stages.train(run=run, trainer=trainer, model=model, data=data)
        representations = tslab.ts.stages.encode(
            run=run,
            trainer=trainer,
            model=model,
            checkpoint=checkpoint,
            data=data,
            encoder=config.encoder,
        )
        tslab.ts.stages.evaluate(
            run=run, representations=representations, downstream_model=SVMProbe()
        )
        for std in config.noise_levels:
            noisy = add_noise(run=run, name=f'std-{std}', representations=representations, std=std)
            tslab.ts.stages.evaluate(
                run=run, name=f'noise-std-{std}', representations=noisy, downstream_model=SVMProbe()
            )


if __name__ == '__main__':
    main(config=tslab.lab.parse_config(config_class=ClassifyConfig, presets=PRESETS))

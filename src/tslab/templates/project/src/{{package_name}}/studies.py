"""A study: rounds of experiments over the classification and forecasting scripts.

A study is a named mapping of grid name to grid. Each grid runs one script
command over the product of its axes; every run lands in
``runs/<study>/<grid>/<axis values>/``. Run one:

    uv run tslab study submit {{package_name}}.studies:example \
        --machine local --grid classification --only seed=0 --dry-run
"""

from pathlib import Path

from tslab.adapters.chronocratic.datasets import ucr_datasets
from tslab.study import from_presets, Grid, Override, setting_axis, Study

from {{package_name}} import classify, forecast

example = Study(
    name='example',
    grids={
        'classification': Grid(
            short_name='cls',
            command=('uv', 'run', 'python', '-m', '{{package_name}}.classify', 'no-preset'),
            config=classify.ClassifyConfig,
            axes={
                # The recipe: one preset of the script, without its dataset (the
                # dataset axis chooses it) and run path (the study sets run paths).
                'recipe': from_presets(
                    presets=classify.PRESETS,
                    names=('ts2vec-archive-split',),
                    without=('data', 'run_path'),
                ),
                # The dataset: a UCR archive under the data/ folder of the project.
                'dataset': ucr_datasets(root=Path('data'), names=('ToySpectra',)),
                # The seed, one value per run.
                'seed': setting_axis(path='seed', values=(0, 1)),
            },
            # Folders such as classification/ts2vec-archive-split/ToySpectra/seed_0;
            # --only seed=0 still works.
            run_path='{grid}/{recipe}/{dataset}/seed_{seed}',
        ),
        'forecasting': Grid(
            short_name='fc',
            command=('uv', 'run', 'python', '-m', '{{package_name}}.forecast', 'no-preset'),
            config=forecast.ForecastConfig,
            axes={
                'recipe': from_presets(
                    presets=forecast.PRESETS,
                    names=('ts2vec-full-series', 'cost-cropped'),
                    without=('data', 'run_path'),
                ),
                'variant': {
                    'ETTh1': {
                        'data': 'ett',
                        'data.dataset_file_path': 'data/toy_hourly.csv.gz',
                        'data.variant': 'ETTh1',
                    }
                },
                'seed': setting_axis(path='seed', values=(0,)),
            },
            # Folders such as forecasting/cost-cropped/ETTh1/seed_0.
            run_path='{grid}/{recipe}/{variant}/seed_{seed}',
            # The batch size of the preset lived in its data settings, which a study
            # drops; give the run its batch size back here.
            overrides=(
                Override(match={'recipe': 'cost-cropped'}, settings={'data.batch_size': 1}),
            ),
        ),
    },
)

"""The full example study, with classification and forecasting grids."""

from tslab.lab.study import Study

from {{package_name}}.studies.common import classification_grid, forecasting_grid

study = Study(
    name='example',
    settings={
        'trainer.max_epochs': 500,
        'trainer.max_training_time': '6:00:00',
        'trainer.early_stopping_patience': 10,
        'trainer.early_stopping_min_delta': 0.001,
    },
    grids={
        'classification': classification_grid(seeds=(0, 1)),
        'forecasting': forecasting_grid(seeds=(0, 1)),
    },
)

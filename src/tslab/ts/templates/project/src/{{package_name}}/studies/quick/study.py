"""A small classification study for a quick machine canary."""

from tslab.lab.study import Study

from {{package_name}}.studies.common import classification_grid

study = Study(
    name='quick',
    settings={'trainer.max_epochs': 5, 'trainer.max_training_time': '0:10:00'},
    grids={'classification': classification_grid(seeds=(0,))},
)

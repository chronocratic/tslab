"""A chained study: the probe grid runs after the pretrain grid has trained."""

from tslab.lab.study import Study

from {{package_name}}.studies.chain.grids import probe_grid
from {{package_name}}.studies.common import classification_grid

study = Study(
    name='chain',
    settings={'trainer.max_epochs': 5, 'trainer.max_training_time': '0:10:00'},
    grids={'pretrain': classification_grid(seeds=(0,)), 'probe': probe_grid(seeds=(0,))},
)

"""``TensorBoardTracker``: TensorBoard event files under ``trackers/tensorboard``."""

from pathlib import Path
from typing import override

from lightning.pytorch.loggers import Logger, TensorBoardLogger

from tslab.tracking.lightning_bridge import LightningLoggerTracker

__all__ = ['TensorBoardTracker']


class TensorBoardTracker(LightningLoggerTracker):
    """Wraps Lightning's ``TensorBoardLogger``. Needs the ``tslab[tensorboard]`` extra."""

    @override
    def _create_logger(self, *, run_directory: Path, run_name: str) -> Logger | None:
        # default_hp_metric=False: without it TensorBoard shows a placeholder "hp_metric = -1".
        return TensorBoardLogger(
            save_dir=run_directory / 'trackers',
            name='tensorboard',
            version='',
            default_hp_metric=False,
        )

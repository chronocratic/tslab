"""``CSVTracker``: metrics and parameters as local CSV/YAML files. Always on."""

from pathlib import Path
from typing import override

from lightning.pytorch.loggers import CSVLogger, Logger

from tslab.tracking.lightning_bridge import LightningLoggerTracker

__all__ = ['CSVTracker']


class CSVTracker(LightningLoggerTracker):
    """Writes ``trackers/csv/version_<n>/metrics.csv`` and ``hparams.yaml``.

    It needs no service and no extra package, so ``Run`` always adds it: it is
    the fallback record when every remote tracker is down.

    Each execution of the run gets the next ``version_<n>`` folder. A fixed
    folder would lose data: Lightning's CSV writer deletes an existing
    ``metrics.csv`` when it opens, so a resumed run would erase the metrics of
    the interrupted one.
    """

    @override
    def _create_logger(self, *, run_directory: Path, run_name: str) -> Logger | None:
        return CSVLogger(save_dir=run_directory / 'trackers', name='csv')

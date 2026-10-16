"""The experiment-tracker interface (W&B, MLflow, TensorBoard, CSV)."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path

__all__ = ['RunStatus', 'Tracker']


class RunStatus(StrEnum):
    """Final status reported to a tracker when the run ends."""

    FINISHED = 'finished'
    FAILED = 'failed'
    CRASHED = 'crashed'


class Tracker(ABC):
    """Sends parameters, metrics and files of one run to an experiment tracker.

    ``Run`` calls ``start`` once on enter (the run folder is only known then),
    the ``log_*`` methods while the run is active, and ``finish`` once on exit.
    ``Run`` catches and logs every exception a tracker raises, so a tracker
    outage never stops the experiment.
    """

    @abstractmethod
    def start(self, *, run_directory: Path, run_name: str) -> None:
        """Open the tracker run.

        Args:
            run_directory: The run folder; trackers write under ``run_directory / "trackers"``.
            run_name: The run's relative path in POSIX form, for display.
        """

    @abstractmethod
    def log_params(self, *, params: Mapping[str, object]) -> None:
        """Record flat configuration parameters (keys joined with ``/``)."""

    @abstractmethod
    def log_metrics(self, *, metrics: Mapping[str, float], step: int | None = None) -> None:
        """Record scalar metrics, optionally at a step."""

    @abstractmethod
    def log_file(self, *, path: Path) -> None:
        """Upload or copy a file to the tracker run."""

    @abstractmethod
    def finish(self, *, status: RunStatus) -> None:
        """Close the tracker run with its final status."""

"""Options that select which trackers a run uses."""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from tslab.lab.core.markers import MachineSetting
from tslab.ml.tracking.wandb_mode import WandbMode

__all__ = ['TrackingOptions']


@dataclass(frozen=True, kw_only=True)
class TrackingOptions:
    """Which trackers a run uses (``--tracking.*`` flags via tyro).

    The CSV tracker is not listed: ``Run`` always adds it.

    Attributes:
        wandb_mode: online, offline, or disabled.
        wandb_project: W&B project.
        wandb_entity: W&B entity (user or team); unset uses your default entity.
        wandb_tags: Extra W&B tags; the parts of the run path are always added.
        mlflow: Enable MLflow.
        mlflow_tracking_uri: MLflow tracking URI; unset uses a SQLite file in the run folder.
        mlflow_experiment: MLflow experiment name; unset uses ``wandb_project``.
        tensorboard: Enable TensorBoard.
        shell_log_path: SLURM stdout/stderr file uploaded to W&B at the end of the run.
    """

    wandb_mode: Annotated[WandbMode, MachineSetting] = WandbMode.DISABLED
    wandb_project: str = 'tslab'
    wandb_entity: str | None = None
    wandb_tags: tuple[str, ...] = ()
    mlflow: bool = False
    mlflow_tracking_uri: str | None = None
    mlflow_experiment: str | None = None
    tensorboard: bool = False
    shell_log_path: Path | None = None

    def fingerprint_settings(self) -> None:
        """Tracking never changes results, so it never enters a fingerprint."""
        return

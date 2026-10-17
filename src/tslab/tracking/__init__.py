"""Experiment tracking: options, the ready trackers and ``build_trackers``.

This module stays importable without Lightning or any tracker SDK: the
concrete trackers are imported inside ``build_trackers``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tslab.interfaces.tracker import Tracker

__all__ = ['TrackingOptions', 'build_trackers']


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

    wandb_mode: Literal['online', 'offline', 'disabled'] = 'disabled'
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


def build_trackers(*, options: TrackingOptions) -> tuple[Tracker, ...]:
    """Build the trackers ``options`` enables (CSV is added by ``Run`` itself).

    Args:
        options: The tracking options, usually ``config.tracking``.

    Returns:
        Unstarted trackers; ``Run`` starts and finishes them.
    """
    # why: imported here so that `import tslab.tracking` never imports Lightning
    # or any tracker SDK; the bare module must stay cheap to import.
    from tslab.tracking.mlflow_tracker import MLflowTracker  # noqa: PLC0415
    from tslab.tracking.tensorboard_tracker import TensorBoardTracker  # noqa: PLC0415
    from tslab.tracking.wandb_tracker import WandbTracker  # noqa: PLC0415

    trackers: list[Tracker] = []
    if options.wandb_mode != 'disabled':
        trackers.append(
            WandbTracker(
                project=options.wandb_project,
                entity=options.wandb_entity,
                mode=options.wandb_mode,
                tags=options.wandb_tags,
                shell_log_path=options.shell_log_path,
            )
        )
    if options.mlflow:
        trackers.append(
            MLflowTracker(
                experiment=options.mlflow_experiment or options.wandb_project,
                tracking_uri=options.mlflow_tracking_uri,
            )
        )
    if options.tensorboard:
        trackers.append(TensorBoardTracker())
    return tuple(trackers)

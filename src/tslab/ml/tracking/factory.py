"""Build only the tracking integrations selected by run options."""

from tslab.lab.core.tracker import Tracker
from tslab.ml.tracking.options import TrackingOptions
from tslab.ml.tracking.wandb_mode import WandbMode

__all__ = ['build_trackers']


def build_trackers(*, options: TrackingOptions) -> tuple[Tracker, ...]:
    """Build the trackers ``options`` enables (CSV is added by ``Run`` itself).

    Args:
        options: The tracking options, usually ``config.tracking``.

    Returns:
        Unstarted trackers; ``Run`` starts and finishes them.
    """
    trackers: list[Tracker] = []
    # why: imported inside each branch so that `import tslab.ml.tracking` never imports
    # Lightning or a tracker SDK, and a tracker nobody enabled is never imported.
    if options.wandb_mode is not WandbMode.DISABLED:
        from tslab.ml.tracking.wandb.tracker import WandbTracker  # noqa: PLC0415

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
        from tslab.ml.tracking.mlflow_tracker import MLflowTracker  # noqa: PLC0415

        trackers.append(
            MLflowTracker(
                experiment=options.mlflow_experiment or options.wandb_project,
                tracking_uri=options.mlflow_tracking_uri,
            )
        )
    if options.tensorboard:
        from tslab.ml.tracking.tensorboard_tracker import TensorBoardTracker  # noqa: PLC0415

        trackers.append(TensorBoardTracker())
    return tuple(trackers)

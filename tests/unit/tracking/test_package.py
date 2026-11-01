from tslab.tracking import build_trackers, TrackingOptions
from tslab.tracking.mlflow_tracker import MLflowTracker
from tslab.tracking.tensorboard_tracker import TensorBoardTracker
from tslab.tracking.wandb_tracker import WandbTracker


def test_build_trackers_follows_options() -> None:
    assert build_trackers(options=TrackingOptions()) == ()
    trackers = build_trackers(
        options=TrackingOptions(wandb_mode='offline', mlflow=True, tensorboard=True)
    )
    assert [type(tracker) for tracker in trackers] == [
        WandbTracker,
        MLflowTracker,
        TensorBoardTracker,
    ]


def test_tracking_options_never_enter_fingerprints() -> None:
    assert TrackingOptions(wandb_mode='online').fingerprint_settings() is None

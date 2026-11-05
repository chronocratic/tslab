from dataclasses import dataclass, field

from tslab.core.config import config_from_yaml, config_to_yaml, parse_config
from tslab.enums import WandbMode
from tslab.tracking import build_trackers, TrackingOptions
from tslab.tracking.mlflow_tracker import MLflowTracker
from tslab.tracking.tensorboard_tracker import TensorBoardTracker
from tslab.tracking.wandb.tracker import WandbTracker


def test_build_trackers_follows_options() -> None:
    assert build_trackers(options=TrackingOptions()) == ()
    trackers = build_trackers(
        options=TrackingOptions(wandb_mode=WandbMode.OFFLINE, mlflow=True, tensorboard=True)
    )
    assert [type(tracker) for tracker in trackers] == [
        WandbTracker,
        MLflowTracker,
        TensorBoardTracker,
    ]


def test_package_exports_are_defined_in_sibling_modules() -> None:
    assert TrackingOptions.__module__ == 'tslab.tracking.options'
    assert build_trackers.__module__ == 'tslab.tracking.factory'


def test_tracking_options_never_enter_fingerprints() -> None:
    assert TrackingOptions(wandb_mode=WandbMode.ONLINE).fingerprint_settings() is None


def test_wandb_mode_is_read_from_a_config_file_and_the_command_line() -> None:
    @dataclass(frozen=True, kw_only=True)
    class Holder:
        tracking: TrackingOptions = field(default_factory=TrackingOptions)

    text = config_to_yaml(config=Holder(tracking=TrackingOptions(wandb_mode=WandbMode.OFFLINE)))
    assert 'wandb_mode: offline' in text
    assert config_from_yaml(config_class=Holder, text=text).tracking.wandb_mode is WandbMode.OFFLINE
    parsed = parse_config(config_class=Holder, args=['--tracking.wandb-mode', 'offline'])
    assert parsed.tracking.wandb_mode is WandbMode.OFFLINE

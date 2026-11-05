from pathlib import Path
import typing
from typing import Annotated

import pytest

from tslab.core.fingerprint import is_not_fingerprinted, NotFingerprinted
from tslab.core.markers import (
    HardwareSetting,
    is_hardware_setting,
    is_machine_setting,
    MachineSetting,
)
from tslab.lightning.config import LightningTrainerConfig
from tslab.tracking import TrackingOptions


def test_machine_marker_and_subclass() -> None:
    assert is_machine_setting(annotation=Annotated[int, MachineSetting])
    assert is_machine_setting(annotation=Annotated[int, HardwareSetting])
    assert is_machine_setting(annotation=Annotated[int, HardwareSetting()])
    assert not is_machine_setting(annotation=Annotated[int, NotFingerprinted])
    assert not is_machine_setting(annotation=int)
    assert is_hardware_setting(annotation=Annotated[int, HardwareSetting, NotFingerprinted])
    assert not is_hardware_setting(annotation=Annotated[int, MachineSetting])


def test_marked_fields() -> None:
    hints = typing.get_type_hints(LightningTrainerConfig, include_extras=True)
    for name in ('accelerator', 'devices'):
        assert is_hardware_setting(annotation=hints[name])
    for name in ('job_time_share', 'enable_progress_bar'):
        assert is_machine_setting(annotation=hints[name])
        assert not is_hardware_setting(annotation=hints[name])
    for name in ('max_epochs', 'max_training_time', 'log_every_n_steps', 'checkpoint_save_top_k'):
        assert not is_machine_setting(annotation=hints[name])
    assert is_not_fingerprinted(annotation=hints['accelerator'])
    assert is_not_fingerprinted(annotation=hints['devices'])
    assert LightningTrainerConfig().devices == 1

    tracking = typing.get_type_hints(TrackingOptions, include_extras=True)
    assert is_machine_setting(annotation=tracking['wandb_mode'])
    assert not is_machine_setting(annotation=tracking['wandb_project'])


@pytest.mark.chronocratic
def test_chronocratic_hardware_fields() -> None:
    from tslab.adapters.chronocratic.datasets import data_settings  # noqa: PLC0415 # why: optional extra
    from tslab.adapters.chronocratic.models import ChronocraticEncoder  # noqa: PLC0415 # why: optional extra

    encoder = typing.get_type_hints(ChronocraticEncoder, include_extras=True)
    assert is_hardware_setting(annotation=encoder['device'])
    assert is_hardware_setting(annotation=encoder['num_workers'])

    settings = data_settings(
        name='ucr', dataset_folder_path=Path('data'), target_column_name='target'
    )
    data_hints = typing.get_type_hints(type(settings), include_extras=True)
    assert is_hardware_setting(annotation=data_hints['num_workers'])
    assert is_not_fingerprinted(annotation=data_hints['num_workers'])

from pathlib import Path
import typing

import pytest

from tslab.lab.core.fingerprint import is_not_fingerprinted
from tslab.lab.core.markers import is_hardware_setting


@pytest.mark.chronocratic
def test_chronocratic_hardware_fields() -> None:
    from tslab.ts.adapters.chronocratic.datasets import data_settings  # noqa: PLC0415 # why: optional extra
    from tslab.ts.adapters.chronocratic.models import ChronocraticEncoder  # noqa: PLC0415 # why: optional extra

    encoder = typing.get_type_hints(ChronocraticEncoder, include_extras=True)
    assert is_hardware_setting(annotation=encoder['device'])
    assert is_hardware_setting(annotation=encoder['num_workers'])

    settings = data_settings(
        name='ucr', dataset_folder_path=Path('data'), target_column_name='target'
    )
    data_hints = typing.get_type_hints(type(settings), include_extras=True)
    assert is_hardware_setting(annotation=data_hints['num_workers'])
    assert is_not_fingerprinted(annotation=data_hints['num_workers'])

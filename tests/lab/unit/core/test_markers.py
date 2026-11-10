from typing import Annotated

from tslab.lab.core.fingerprint import NotFingerprinted
from tslab.lab.core.markers import (
    HardwareSetting,
    is_hardware_setting,
    is_machine_setting,
    MachineSetting,
)


def test_machine_marker_and_subclass() -> None:
    assert is_machine_setting(annotation=Annotated[int, MachineSetting])
    assert is_machine_setting(annotation=Annotated[int, HardwareSetting])
    assert is_machine_setting(annotation=Annotated[int, HardwareSetting()])
    assert not is_machine_setting(annotation=Annotated[int, NotFingerprinted])
    assert not is_machine_setting(annotation=int)
    assert is_hardware_setting(annotation=Annotated[int, HardwareSetting, NotFingerprinted])
    assert not is_hardware_setting(annotation=Annotated[int, MachineSetting])

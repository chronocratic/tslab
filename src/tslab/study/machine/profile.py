"""The data classes of a machine profile."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from tslab.study.machine.local import LocalSettings
from tslab.study.machine.slurm import SlurmSettings
from tslab.study.settings import SettingValue

__all__ = ['FollowupSettings', 'MachineOverride', 'MachineProfile']


@dataclass(frozen=True, kw_only=True)
class FollowupSettings:
    """The ``[followup]`` table: the small helper jobs of a study."""

    auto_retry: bool = False
    max_attempts: int = 3
    wandb_sync: bool = False
    time: str = '00:30:00'


@dataclass(frozen=True, kw_only=True)
class MachineOverride:
    """One ``[[overrides]]`` entry: match plus slurm and/or settings."""

    match: dict[str, tuple[str, ...]]
    slurm: dict[str, object] = field(default_factory=dict)
    settings: dict[str, SettingValue] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class MachineProfile:
    """A machine profile read from TOML, fully checked."""

    path: Path
    launcher: Literal['slurm', 'local']
    output_root: Path
    tslab_command: tuple[str, ...] = ('uv', 'run', 'tslab')
    settings: dict[str, SettingValue] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    slurm: SlurmSettings | None = None
    local: LocalSettings | None = None
    followup: FollowupSettings = FollowupSettings()
    overrides: tuple[MachineOverride, ...] = ()

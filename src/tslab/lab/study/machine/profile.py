"""The data classes of a machine profile."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from tslab.lab.enums import ChainedRetry, LauncherKind, Retry
from tslab.lab.study.definition import SettingValue
from tslab.lab.study.machine.local import LocalSettings
from tslab.lab.study.machine.slurm import SlurmSettings

__all__ = ['FollowupSettings', 'MachineOverride', 'MachineProfile', 'wants_retry_followup']


@dataclass(frozen=True, kw_only=True)
class FollowupSettings:
    """The ``[followup]`` table: the small helper jobs of a study.

    Attributes:
        retry: Whether to resubmit grids without dependencies.
        chained_retry: Whether to retry grids that use dependencies.
        max_attempts: The maximum attempts for one run.
        wandb_sync: Whether to queue a W&B synchronization job.
        time: The time limit of follow-up jobs.
    """

    retry: Retry = Retry.MANUAL
    chained_retry: ChainedRetry = ChainedRetry.MANUAL
    max_attempts: int = 3
    wandb_sync: bool = False
    time: str = '00:30:00'


def wants_retry_followup(*, followup: FollowupSettings) -> bool:
    """Whether either retry setting needs a follow-up job.

    Args:
        followup: The follow-up settings of the machine profile.

    Returns:
        Whether any retry mode needs a follow-up job.
    """
    return followup.retry is not Retry.MANUAL or followup.chained_retry is not ChainedRetry.MANUAL


@dataclass(frozen=True, kw_only=True)
class MachineOverride:
    """One ``[[overrides]]`` entry: match plus slurm and/or settings.

    Attributes:
        match: The axis names and the values that select the runs; read-only.
        slurm: The SLURM fields the entry changes; read-only.
        settings: The study settings the entry changes; read-only.
    """

    match: Mapping[str, tuple[str, ...]]
    slurm: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))
    settings: Mapping[str, SettingValue] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True, kw_only=True)
class MachineProfile:
    """A machine profile read from TOML, fully checked.

    Attributes:
        path: The profile file.
        launcher: The backend that runs the study.
        output_root: The folder that holds the study output.
        tslab_command: The command that starts TSLab inside a job.
        settings: The study settings every run gets; read-only.
        env: The environment variables of the jobs; read-only.
        slurm: The SLURM settings, for a SLURM profile.
        local: The local settings, for a local profile.
        followup: The follow-up job settings.
        overrides: The entries that change the settings of selected runs.
    """

    path: Path
    launcher: LauncherKind
    output_root: Path
    tslab_command: tuple[str, ...] = ('uv', 'run', 'tslab')
    settings: Mapping[str, SettingValue] = field(default_factory=lambda: MappingProxyType({}))
    env: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    slurm: SlurmSettings | None = None
    local: LocalSettings | None = None
    followup: FollowupSettings = FollowupSettings()
    overrides: tuple[MachineOverride, ...] = ()

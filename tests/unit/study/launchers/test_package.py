"""The backend table: one backend per launcher, picked from the profile."""

import os
from pathlib import Path

import pytest

from tslab.study.launchers import backend_for, BACKENDS
from tslab.study.machine import (
    load_machine,
    LocalSettings,
    MachineProfile,
    MachineProfileError,
    SlurmSettings,
)


def test_backend_for_picks_by_launcher(*, tmp_path: Path) -> None:
    slurm = MachineProfile(
        path=tmp_path / 'slurm.toml',
        launcher='slurm',
        output_root=tmp_path / 'out',
        slurm=SlurmSettings(partition='gpu', time='01:00:00'),
    )
    local = MachineProfile(
        path=tmp_path / 'local.toml',
        launcher='local',
        output_root=tmp_path / 'out',
        local=LocalSettings(),
    )
    assert backend_for(machine=slurm) is BACKENDS['slurm']
    assert backend_for(machine=local) is BACKENDS['local']


def test_unknown_launcher_is_rejected_by_the_profile(*, tmp_path: Path) -> None:
    path = tmp_path / 'machine.toml'
    path.write_text('launcher = "kubernetes"\noutput_root = "out"\n', encoding='utf-8')
    with pytest.raises(MachineProfileError, match=r'launcher must be "slurm" or "local"'):
        load_machine(path=path, environment=os.environ)

"""The backend table: one backend per launcher, picked from the profile."""

import dataclasses
from datetime import timedelta
import os
from pathlib import Path

import pytest

from tslab.study.launchers import backend_for, BACKENDS, JobUsage, LaunchError
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


def test_launch_error_carries_the_scheduler_message() -> None:
    error = LaunchError('sbatch: error: invalid partition')
    assert isinstance(error, RuntimeError)
    assert str(error) == 'sbatch: error: invalid partition'


def test_job_usage_is_a_frozen_neutral_record() -> None:
    usage = JobUsage(
        state='COMPLETED',
        elapsed=timedelta(minutes=5),
        time_limit=timedelta(hours=1),
        total_cpu=timedelta(seconds=90),
        max_rss=2 * 1024**3,
        alloc_cpus=2,
        gpus=1,
    )
    assert [field.name for field in dataclasses.fields(usage)] == [
        'state',
        'elapsed',
        'time_limit',
        'total_cpu',
        'max_rss',
        'alloc_cpus',
        'gpus',
        'exit_code',
    ]
    assert usage.__dataclass_params__.frozen

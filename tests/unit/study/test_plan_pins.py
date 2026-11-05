from pathlib import Path

import pytest

from tests.unit.study.configs import grid
from tslab.enums import LauncherKind
from tslab.study import Study
from tslab.study.definition import expand_runs
from tslab.study.definition.settings import SettingValue
from tslab.study.machine import MachineProfile
from tslab.study.plan import pin_conflicts, plan_jobs, resolve_runs


def _machine(*, tmp_path: Path, settings: dict[str, SettingValue]) -> MachineProfile:
    return MachineProfile(
        path=tmp_path / 'm.toml',
        launcher=LauncherKind.LOCAL,
        output_root=tmp_path / 'runs',
        settings=settings,
    )


def test_same_value_passes() -> None:
    assert (
        pin_conflicts(
            before={'trainer.devices': 1}, after={'trainer.devices': 1}, env={}, placeholders={}
        )
        == []
    )


def test_different_value_is_reported() -> None:
    assert pin_conflicts(
        before={'trainer.devices': 1}, after={'trainer.devices': 2}, env={}, placeholders={}
    ) == [('trainer.devices', 1, 2)]


def test_values_are_compared_after_expansion() -> None:
    assert (
        pin_conflicts(
            before={'data.num_workers': 2},
            after={'data.num_workers': '{cpus}'},
            env={},
            placeholders={'cpus': '2'},
        )
        == []
    )


def test_plan_stops_on_pin_conflict(tmp_path: Path) -> None:
    study = Study(name='s', settings={'trainer.accelerator': 'cpu'}, grids={'g': grid()})
    machine = _machine(tmp_path=tmp_path, settings={'trainer.accelerator': 'gpu'})
    runs = expand_runs(study=study)
    resolved = resolve_runs(study=study, machine=machine, runs=runs, local_cpus=1)

    with pytest.raises(
        ValueError, match=r"study 's' pins trainer\.accelerator = 'cpu', but .*\.toml sets 'gpu'"
    ):
        plan_jobs(
            study=study,
            machine=machine,
            resolved=resolved,
            indices={run.run_path: index for index, run in enumerate(runs)},
            extra_args={},
            attempts={},
            first_job_file=1,
            environment={},
        )

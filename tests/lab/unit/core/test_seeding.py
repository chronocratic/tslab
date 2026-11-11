from collections.abc import Iterator
import dataclasses
import logging
from pathlib import Path
import random
import sys
import types

import numpy as np
import pytest

from tslab.lab.core import seeding
from tslab.lab.core.meta import run as meta_run
from tslab.lab.core.run import Run
from tslab.lab.core.seeding import register_seeder, seed_everything


@dataclasses.dataclass(frozen=True, kw_only=True)
class Config:
    seed: int = 1


@pytest.fixture(autouse=True)
def _restore_registry() -> Iterator[None]:
    saved = dict(seeding._seeders)  # noqa: SLF001
    warned = seeding._warned  # noqa: SLF001
    last = seeding._last_seed  # noqa: SLF001
    yield
    seeding._seeders.clear()  # noqa: SLF001
    seeding._seeders.update(saved)  # noqa: SLF001
    seeding._warned = warned  # noqa: SLF001
    seeding._last_seed = last  # noqa: SLF001


def test_seed_everything_repeats_python_and_numpy() -> None:
    def draw() -> tuple[float, float]:
        return (
            random.random(),  # noqa: S311 # why: not cryptography; checks seeding
            float(np.random.rand()),  # noqa: NPY002 # why: the legacy global generator is under test
        )

    seed_everything(seed=3)
    first = draw()
    seed_everything(seed=3)
    assert draw() == first
    seed_everything(seed=4)
    assert draw() != first


def test_registered_seeder_runs_once_with_the_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(seeding, '_last_seed', None)
    seen: list[int] = []
    register_seeder(seen.append, name='seen')
    register_seeder(seen.append, name='seen')
    seed_everything(seed=7)
    assert seen[-1] == 7
    assert seen.count(7) == 1


def test_registering_a_name_again_replaces_the_seeder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(seeding, '_last_seed', None)
    first: list[int] = []
    second: list[int] = []
    register_seeder(first.append, name='replaced')
    register_seeder(second.append, name='replaced')
    seed_everything(seed=5)
    assert first == []
    assert second[-1] == 5


def test_a_seeder_registered_after_seeding_gets_the_last_seed() -> None:
    seed_everything(seed=11)
    late: list[int] = []
    register_seeder(late.append, name='late_seeder')
    assert late == [11]


def test_warns_once_when_torch_is_loaded_and_no_seeder_is_registered(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setitem(sys.modules, 'torch', types.ModuleType('torch'))
    monkeypatch.setattr(seeding, '_seeders', {})
    monkeypatch.setattr(seeding, '_warned', False)
    with caplog.at_level(logging.WARNING, logger=seeding.__name__):
        seed_everything(seed=1)
        seed_everything(seed=1)
    assert [r.getMessage() for r in caplog.records] == [
        'torch is loaded but not seeded; import tslab.ml before the run starts to seed it'
    ]


def test_a_seeder_that_is_not_torch_still_warns(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(seeding, '_warned', False)
    monkeypatch.setitem(sys.modules, 'torch', types.ModuleType('torch'))
    monkeypatch.setattr(seeding, '_seeders', {'jax': lambda _seed: None})
    with caplog.at_level(logging.WARNING, logger=seeding.__name__):
        seed_everything(seed=1)
    assert len(caplog.records) == 1


def test_no_warning_with_a_torch_seeder_or_without_torch(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(seeding, '_warned', False)
    monkeypatch.setitem(sys.modules, 'torch', types.ModuleType('torch'))
    monkeypatch.setattr(seeding, '_seeders', {'torch': lambda _seed: None})
    with caplog.at_level(logging.WARNING, logger=seeding.__name__):
        seed_everything(seed=1)
    monkeypatch.delitem(sys.modules, 'torch')
    monkeypatch.setattr(seeding, '_seeders', {})
    with caplog.at_level(logging.WARNING, logger=seeding.__name__):
        seed_everything(seed=1)
    assert caplog.records == []


def test_collecting_run_metadata_warns_when_torch_arrives_unseeded(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    monkeypatch.setattr(seeding, '_warned', False)
    monkeypatch.setattr(seeding, '_seeders', {})
    monkeypatch.setattr(meta_run, '_version_collectors', {})
    monkeypatch.setattr(meta_run, '_hardware_collectors', {})
    run = Run(config=Config(), root=tmp_path, relative_path=Path('a'))
    monkeypatch.setitem(sys.modules, 'torch', types.ModuleType('torch'))
    with caplog.at_level(logging.WARNING, logger=seeding.__name__):
        run.environment()
    assert len(caplog.records) == 1

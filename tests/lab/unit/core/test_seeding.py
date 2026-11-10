from collections.abc import Iterator
import logging
import random
import sys
import types

import numpy as np
import pytest

from tslab.lab.core import seeding
from tslab.lab.core.seeding import register_seeder, seed_everything


@pytest.fixture(autouse=True)
def _restore_registry() -> Iterator[None]:
    saved = list(seeding._seeders)  # noqa: SLF001
    warned = seeding._warned  # noqa: SLF001
    yield
    seeding._seeders[:] = saved  # noqa: SLF001
    seeding._warned = warned  # noqa: SLF001


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


def test_registered_seeder_runs_once_with_the_seed() -> None:
    seen: list[int] = []
    register_seeder(seen.append)
    register_seeder(seen.append)
    seed_everything(seed=7)
    assert seen == [7]


def test_warns_once_when_torch_is_loaded_and_no_seeder_is_registered(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setitem(sys.modules, 'torch', types.ModuleType('torch'))
    monkeypatch.setattr(seeding, '_seeders', [])
    monkeypatch.setattr(seeding, '_warned', False)
    with caplog.at_level(logging.WARNING, logger=seeding.__name__):
        seed_everything(seed=1)
        seed_everything(seed=1)
    assert [r.getMessage() for r in caplog.records] == [
        'torch is loaded but not seeded; import tslab.ml before the run starts to seed it'
    ]


def test_no_warning_with_a_seeder_or_without_torch(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(seeding, '_warned', False)
    monkeypatch.setitem(sys.modules, 'torch', types.ModuleType('torch'))
    monkeypatch.setattr(seeding, '_seeders', [lambda _seed: None])
    with caplog.at_level(logging.WARNING, logger=seeding.__name__):
        seed_everything(seed=1)
    monkeypatch.delitem(sys.modules, 'torch')
    monkeypatch.setattr(seeding, '_seeders', [])
    with caplog.at_level(logging.WARNING, logger=seeding.__name__):
        seed_everything(seed=1)
    assert caplog.records == []

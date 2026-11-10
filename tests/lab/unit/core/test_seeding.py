from collections.abc import Iterator
import random

import numpy as np
import pytest

from tslab.lab.core import seeding
from tslab.lab.core.seeding import register_seeder, seed_everything


@pytest.fixture(autouse=True)
def _restore_registry() -> Iterator[None]:
    saved = list(seeding._seeders)  # noqa: SLF001
    yield
    seeding._seeders[:] = saved  # noqa: SLF001


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

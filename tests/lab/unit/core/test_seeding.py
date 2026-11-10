import random

import numpy as np
import torch

from tslab.lab.core.seeding import seed_everything


def draw() -> tuple[float, float, float]:
    # The global generators are exactly what seed_everything must seed.
    return (
        random.random(),  # noqa: S311 # why: not cryptography; checks seeding
        float(np.random.rand()),  # noqa: NPY002 # why: the legacy global generator is under test
        float(torch.rand(1)),
    )


def test_seed_everything_repeats_all_generators() -> None:
    seed_everything(seed=3)
    first = draw()
    seed_everything(seed=3)
    assert draw() == first
    seed_everything(seed=4)
    assert draw() != first

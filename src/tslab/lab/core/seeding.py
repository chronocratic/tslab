"""Seeding of every random number generator TSLab knows about.

``seed_everything`` seeds ``random`` and NumPy. Other layers add their own
generators with ``register_seeder`` when they are imported.
"""

from collections.abc import Callable
import os
import random

import numpy as np

__all__ = ['register_seeder', 'seed_everything']

_seeders: list[Callable[[int], None]] = []


def register_seeder(seeder: Callable[[int], None]) -> None:
    """Add a function that seeds one more generator; registering it twice is a no-op.

    Args:
        seeder: Called with the seed by every ``seed_everything`` call, in registration order.
    """
    if seeder not in _seeders:
        _seeders.append(seeder)


def seed_everything(*, seed: int) -> None:
    """Seed the ``random`` module of Python, NumPy and every registered generator.

    ``PYTHONHASHSEED`` is also set so that subprocesses (for example data-loader
    workers started with ``spawn``) hash strings identically.

    Args:
        seed: The seed. The same seed gives the same random streams.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    # why: seeds the legacy global generator many libraries still use
    np.random.seed(seed)  # noqa: NPY002
    for seeder in _seeders:
        seeder(seed)

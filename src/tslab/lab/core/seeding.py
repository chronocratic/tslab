"""Seeding of every random number generator TSLab knows about.

``seed_everything`` seeds ``random`` and NumPy. Other layers add their own
generators with ``register_seeder`` when they are imported. A process that has
loaded torch but registered no ``torch`` seeder gets one warning, at seeding time and
when run metadata is collected.
"""

from collections.abc import Callable
import logging
import os
import random
import sys

import numpy as np

__all__ = ['register_seeder', 'seed_everything', 'warn_if_torch_unseeded']

_LOGGER = logging.getLogger(__name__)
_seeders: dict[str, Callable[[int], None]] = {}
_warned = False
_last_seed: int | None = None


def register_seeder(seeder: Callable[[int], None], *, name: str) -> None:
    """Add a function that seeds one more generator; the same ``name`` replaces the earlier one.

    A seeder registered after ``seed_everything`` ran is called at once with the last seed.

    Args:
        seeder: Called with the seed by every ``seed_everything`` call, in registration order.
        name: What the seeder seeds, for example ``torch``; the torch warning looks for ``torch``.
    """
    _seeders[name] = seeder
    if _last_seed is not None:
        seeder(_last_seed)


def seed_everything(*, seed: int) -> None:
    """Seed the ``random`` module of Python, NumPy and every registered generator.

    ``PYTHONHASHSEED`` is also set so that subprocesses (for example data-loader
    workers started with ``spawn``) hash strings identically.

    Args:
        seed: The seed. The same seed gives the same random streams.
    """
    global _last_seed  # noqa: PLW0603 # why: late seeders get the seed
    _last_seed = seed
    warn_if_torch_unseeded()
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    # why: seeds the legacy global generator many libraries still use
    np.random.seed(seed)  # noqa: NPY002
    for seeder in _seeders.values():
        seeder(seed)


def warn_if_torch_unseeded() -> None:
    """Warn once per process when torch is loaded but no seeder named ``torch`` is registered."""
    global _warned  # noqa: PLW0603 # why: one warning per process
    if _warned or 'torch' in _seeders or 'torch' not in sys.modules:
        return
    _warned = True
    _LOGGER.warning(
        'torch is loaded but not seeded; import tslab.ml before the run starts to seed it'
    )

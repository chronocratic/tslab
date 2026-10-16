"""Seeding of every random number generator TSLab knows about.

This module deliberately does not import Lightning, so the core stays free of
it. Lightning's own ``seed_everything`` does the same work for training code.
"""

import os
import random

import numpy as np
import torch

__all__ = ['seed_everything']


def seed_everything(*, seed: int) -> None:
    """Seed Python's ``random``, NumPy and PyTorch (all devices).

    ``PYTHONHASHSEED`` is also set so that subprocesses (for example data-loader
    workers started with ``spawn``) hash strings identically.

    Args:
        seed: The seed. The same seed gives the same random streams.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 # why: seeds the legacy global generator many libraries still use
    torch.manual_seed(seed)

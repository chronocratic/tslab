"""Choose the checkpoint a training resumes from.

Every candidate is loaded first, the first usable one is returned, and the
trainer calls ``fit`` exactly once. Retrying ``trainer.fit`` from the next
checkpoint whenever ``fit`` raises one of the "corrupt checkpoint" exceptions
would also retry errors raised during training, such as CUDA out-of-memory,
so a real failure would repeat once per checkpoint.

Rollback, if this ever proves worse in practice: wrap each
``trainer.fit(ckpt_path=candidate)`` in ``try/except CHECKPOINT_LOAD_ERRORS``,
walking ``resume_candidates(...)`` and ending with ``fit`` from scratch. Only
this module and ``LightningTrainer.fit`` change.
"""

import logging
from pathlib import Path
import pickle
import re

from lightning.pytorch.utilities.exceptions import MisconfigurationException
import torch

__all__ = ['CHECKPOINT_LOAD_ERRORS', 'choose_resume_checkpoint', 'resume_candidates']

_LOGGER = logging.getLogger(__name__)
_REQUIRED_KEYS = frozenset({'state_dict', 'loops'})

# Errors a damaged or partial checkpoint raises while loading.
CHECKPOINT_LOAD_ERRORS: tuple[type[Exception], ...] = (
    EOFError,
    RuntimeError,
    KeyError,
    OSError,
    ValueError,
    MisconfigurationException,
    pickle.UnpicklingError,
)


def _modified(path: Path) -> float:
    return path.stat().st_mtime


def resume_candidates(*, directory: Path, monitor: str) -> list[Path]:
    """Return the checkpoint files in the order they are tried.

    Args:
        directory: The checkpoint folder.
        monitor: The monitored metric whose value is part of the file names.

    Returns:
        ``last*.ckpt`` newest first, then files named with ``<monitor>=<value>``
        lowest value first, then all other ``.ckpt`` files newest first.
    """
    pattern = re.compile(rf'{re.escape(monitor)}=(-?\d+(?:\.\d+)?)')
    last: list[Path] = []
    ranked: list[tuple[float, Path]] = []
    rest: list[Path] = []
    for path in sorted(directory.glob('*.ckpt')):
        if not path.is_file():
            continue
        match = pattern.search(path.stem)
        if path.name.startswith('last'):
            last.append(path)
        elif match is not None:
            ranked.append((float(match[1]), path))
        else:
            rest.append(path)
    return [
        *sorted(last, key=_modified, reverse=True),
        *(path for _, path in sorted(ranked)),
        *sorted(rest, key=_modified, reverse=True),
    ]


def _unusable_reason(*, path: Path) -> str | None:
    """Return why ``path`` cannot be resumed from, or ``None`` if it can."""
    try:
        # The run's own checkpoint folder: full unpickling is trusted here.
        state = torch.load(f=path, map_location='cpu', weights_only=False)
    except CHECKPOINT_LOAD_ERRORS as error:
        return f'{type(error).__name__}: {error}'
    if not isinstance(state, dict):
        return 'not a Lightning checkpoint'
    missing = sorted(_REQUIRED_KEYS - set(state))
    if missing:
        return f'missing keys {missing}'
    return None


def choose_resume_checkpoint(*, directory: Path, monitor: str) -> Path | None:
    """Return the first usable checkpoint, or ``None`` to train from scratch.

    Args:
        directory: The checkpoint folder.
        monitor: The monitored metric whose value is part of the file names.

    Returns:
        The checkpoint to pass as ``ckpt_path``, or ``None``.
    """
    candidates = resume_candidates(directory=directory, monitor=monitor)
    for candidate in candidates:
        reason = _unusable_reason(path=candidate)
        if reason is None:
            _LOGGER.info('resuming training from %s', candidate.name)
            return candidate
        _LOGGER.warning(
            'cannot resume from %s (%s); trying the next checkpoint', candidate.name, reason
        )
    if candidates:
        _LOGGER.warning('no usable checkpoint in %s; training from scratch', directory)
    return None

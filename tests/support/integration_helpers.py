"""Helpers the integration tests share."""

from collections.abc import Generator
import contextlib
from dataclasses import dataclass, field
from pathlib import Path

import tslab.lab
from tslab.lab.core.tracker import Tracker


@dataclass(frozen=True, kw_only=True)
class Config:
    seed: int = 0
    run: tslab.lab.RunOptions = field(default_factory=tslab.lab.RunOptions)


@contextlib.contextmanager
def run_in(*, root: Path, trackers: tuple[Tracker, ...] = ()) -> Generator[tslab.lab.Run]:
    with tslab.lab.Run(
        config=Config(), root=root, relative_path=Path('r'), trackers=trackers
    ) as run:
        yield run

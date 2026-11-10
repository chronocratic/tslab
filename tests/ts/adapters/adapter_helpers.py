"""Helpers the adapter tests share."""

from collections.abc import Generator
import contextlib
from pathlib import Path

import tslab.lab


def parse[T](config_class: type[T], *args: str) -> T:
    return tslab.lab.parse_config(config_class=config_class, args=list(args))


@contextlib.contextmanager
def run_in(*, root: Path, config: object, name: str = 'r') -> Generator[tslab.lab.Run]:
    with tslab.lab.Run(config=config, root=root, relative_path=Path(name)) as run:
        yield run

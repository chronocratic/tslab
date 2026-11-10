"""Helpers the adapter tests share."""

from collections.abc import Generator
import contextlib
from pathlib import Path

import tslab


def parse[T](config_class: type[T], *args: str) -> T:
    return tslab.parse_config(config_class=config_class, args=list(args))


@contextlib.contextmanager
def run_in(*, root: Path, config: object, name: str = 'r') -> Generator[tslab.Run]:
    with tslab.Run(config=config, root=root, relative_path=Path(name)) as run:
        yield run

# expect-error: none
"""``tslab.REQUIRED`` type-checks where the marked field's annotation stands."""

from dataclasses import dataclass
from pathlib import Path

import tslab


@dataclass(frozen=True, kw_only=True)
class Data:
    folder: Path


def preset() -> Data:
    return Data(folder=tslab.REQUIRED)

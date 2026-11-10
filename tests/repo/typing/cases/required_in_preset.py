# expect-error: none
"""``tslab.lab.REQUIRED`` type-checks where the marked field's annotation stands."""

from dataclasses import dataclass
from pathlib import Path

import tslab.lab


@dataclass(frozen=True, kw_only=True)
class Data:
    folder: Path


def preset() -> Data:
    return Data(folder=tslab.lab.REQUIRED)

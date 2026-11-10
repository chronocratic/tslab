# expect-error: unknown-argument
from pathlib import Path

import tslab.lab


@tslab.lab.stage(name='scale', output=tslab.lab.Json)
def scale(*, run: tslab.lab.Run, factor: float) -> tslab.lab.Json:
    return tslab.lab.Json(data={'factor': factor})


def call(run: tslab.lab.Run) -> None:
    scale(run=run, factr=2.0)
    _ = Path()

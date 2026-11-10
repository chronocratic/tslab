# expect-error: invalid-argument-type
import tslab.lab


@tslab.lab.stage(name='scale', output=tslab.lab.Json)
def scale(*, run: tslab.lab.Run, factor: float) -> tslab.lab.Json:
    return tslab.lab.Json(data={'factor': factor})


def call(run: tslab.lab.Run) -> None:
    scale(run=run, factor='two')

# expect-error: invalid-argument-type
import tslab


@tslab.stage(name='scale', output=tslab.Json)
def scale(*, run: tslab.Run, factor: float) -> tslab.Json:
    return tslab.Json(data={'factor': factor})


def call(run: tslab.Run) -> None:
    scale(run=run, factor='two')

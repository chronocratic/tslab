# expect-error: unresolved-attribute
import tslab.lab


@tslab.lab.stage(name='scale', output=tslab.lab.Json)
def scale(*, run: tslab.lab.Run, factor: float) -> tslab.lab.Json:
    return tslab.lab.Json(data={'factor': factor})


def call(run: tslab.lab.Run) -> None:
    result = scale(run=run, factor=2.0)
    _ = result.tensors

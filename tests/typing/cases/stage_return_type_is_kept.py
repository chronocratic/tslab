# expect-error: unresolved-attribute
import tslab


@tslab.stage(name='scale', output=tslab.Json)
def scale(*, run: tslab.Run, factor: float) -> tslab.Json:
    return tslab.Json(data={'factor': factor})


def call(run: tslab.Run) -> None:
    result = scale(run=run, factor=2.0)
    _ = result.tensors

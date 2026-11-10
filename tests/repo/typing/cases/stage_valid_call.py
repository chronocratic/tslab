# expect-error: none
import tslab


@tslab.stage(name='scale', output=tslab.Json)
def scale(*, run: tslab.Run, name: tslab.InstanceName | None = None, factor: float) -> tslab.Json:
    return tslab.Json(data={'factor': factor})


def call(run: tslab.Run) -> dict[str, object]:
    return scale(run=run, name=('a', 'b'), factor=2.0).data

# expect-error: none
import tslab.lab


@tslab.lab.stage(name='scale', output=tslab.lab.Json)
def scale(*, run: tslab.lab.Run, name: tslab.lab.InstanceName | None = None, factor: float) -> tslab.lab.Json:
    return tslab.lab.Json(data={'factor': factor})


def call(run: tslab.lab.Run) -> dict[str, object]:
    return scale(run=run, name=('a', 'b'), factor=2.0).data

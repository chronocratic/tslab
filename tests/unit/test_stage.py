from dataclasses import dataclass
import json
from pathlib import Path
from typing import Annotated, override

import pytest
import torch

import tslab


@dataclass(frozen=True)
class Config:
    seed: int = 0


calls: list[str] = []


def open_run(tmp_path: Path) -> tslab.Run:
    return tslab.Run(
        config=Config(), root=tmp_path, relative_path=Path('r'), options=tslab.RunOptions()
    )


@tslab.stage(name='count', output=tslab.Json)
def count(
    *, run: tslab.Run, value: int, workers: Annotated[int, tslab.NotFingerprinted] = 0
) -> tslab.Json:
    calls.append(f'count {value} {workers}')
    return tslab.Json(data={'value': value})


def test_not_fingerprinted_argument_does_not_trigger_rerun(tmp_path: Path) -> None:
    calls.clear()
    with open_run(tmp_path) as run:
        count(run=run, value=1, workers=0)
    with open_run(tmp_path) as run:
        count(run=run, value=1, workers=8)
    assert calls == ['count 1 0']


def test_version_bump_reruns(tmp_path: Path) -> None:
    @tslab.stage(name='versioned', output=tslab.Json, version=1)
    def first(*, run: tslab.Run) -> tslab.Json:
        calls.append('v1')
        return tslab.Json(data={})

    @tslab.stage(name='versioned', output=tslab.Json, version=2)
    def second(*, run: tslab.Run) -> tslab.Json:
        calls.append('v2')
        return tslab.Json(data={})

    calls.clear()
    with open_run(tmp_path) as run:
        first(run=run)
    with open_run(tmp_path) as run:
        second(run=run)
    assert calls == ['v1', 'v2']


def test_unfingerprintable_argument_names_the_argument(tmp_path: Path) -> None:
    @tslab.stage(name='bad', output=tslab.Json)
    def bad(*, run: tslab.Run, model: torch.nn.Module) -> tslab.Json:
        return tslab.Json(data={})

    with (
        open_run(tmp_path) as run,
        pytest.raises(TypeError, match=r"bad\[default\]: cannot fingerprint argument 'model'"),
    ):
        bad(run=run, model=torch.nn.Linear(1, 1))


def test_wrong_return_type_is_a_stage_failure(tmp_path: Path) -> None:
    @tslab.stage(name='wrong', output=tslab.Tensors)
    def wrong(*, run: tslab.Run) -> tslab.Tensors:
        return tslab.Json(data={})  # ty: ignore[invalid-return-type]

    with open_run(tmp_path) as run:
        result = wrong(run=run)
    assert not tslab.is_available(artifact=result)
    status = json.loads((tmp_path / 'r' / 'status.json').read_text(encoding='utf-8'))
    assert (
        status['instances']['wrong[default]']['error']
        == "TypeError: stage 'wrong' must return Tensors, got Json"
    )


def test_decorator_requires_keyword_only_and_run() -> None:
    # why: the parameter exists only to be rejected
    with pytest.raises(TypeError, match='keyword-only'):
        tslab.stage(name='positional', output=tslab.Json)(
            lambda run: tslab.Json(data={})  # noqa: ARG005
        )
    with pytest.raises(TypeError, match='must take a run parameter'):

        @tslab.stage(name='norun', output=tslab.Json)
        def no_run(*, value: int) -> tslab.Json:
            return tslab.Json(data={})


def test_corrupt_cached_result_is_recomputed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    calls.clear()
    with open_run(tmp_path) as run:
        count(run=run, value=2)
    (tmp_path / 'r' / 'count' / 'default' / 'data.json').write_text('{broken', encoding='utf-8')
    with open_run(tmp_path) as run:
        result = count(run=run, value=2)
    assert calls == ['count 2 0', 'count 2 0']
    assert 'cannot load count[default]' in caplog.text
    assert isinstance(result, tslab.Json)


def test_corrupt_instance_meta_is_recomputed(tmp_path: Path) -> None:
    calls.clear()
    with open_run(tmp_path) as run:
        count(run=run, value=2)
    (tmp_path / 'r' / 'count' / 'default' / 'meta.json').write_text('not json', encoding='utf-8')
    with open_run(tmp_path) as run:
        result = count(run=run, value=2)
    assert calls == ['count 2 0', 'count 2 0']
    assert isinstance(result, tslab.Json)
    (tmp_path / 'r' / 'count' / 'default' / 'meta.json').write_bytes(b'\xff\xfe\x00\x01')
    with open_run(tmp_path) as run:
        result = count(run=run, value=2)
    assert calls == ['count 2 0', 'count 2 0', 'count 2 0']
    assert isinstance(result, tslab.Json)


def test_external_input_is_recorded_and_content_sensitive(tmp_path: Path) -> None:
    @tslab.stage(name='use', output=tslab.Json)
    def use(*, run: tslab.Run, thresholds: tslab.Json) -> tslab.Json:
        calls.append('use')
        return tslab.Json(data=thresholds.data)

    source = tmp_path / 'thresholds.json'
    source.write_text('{"t": 1}', encoding='utf-8')
    calls.clear()
    for _ in range(2):
        with open_run(tmp_path) as run:
            use(run=run, thresholds=tslab.Json.external(path=source))
    source.write_text('{"t": 2}', encoding='utf-8')
    with open_run(tmp_path) as run:
        use(run=run, thresholds=tslab.Json.external(path=source))
    assert calls == ['use', 'use']
    meta = json.loads(
        (tmp_path / 'r' / 'use' / 'default' / 'meta.json').read_text(encoding='utf-8')
    )
    assert meta['external_inputs']['thresholds']['path'] == source.as_posix()
    assert meta['inputs'] == {}


class Doubler(tslab.Stage[tslab.Json]):
    name = 'double'
    output_type = tslab.Json

    @override
    def execute(  # ty: ignore[invalid-method-override]
        self, *, run: tslab.Run, value: int
    ) -> tslab.Json:
        calls.append('double')
        return tslab.Json(data={'value': 2 * value})


def test_class_based_stage(tmp_path: Path) -> None:
    calls.clear()
    for _ in range(2):
        with open_run(tmp_path) as run:
            result = Doubler()(run=run, value=3)
    assert calls == ['double']
    assert isinstance(result, tslab.Json)
    assert result.data == {'value': 6}


class SimulatedTimeLimitError(RuntimeError):
    pass


def test_in_place_stage_resumes_and_restarts_on_fingerprint_change(tmp_path: Path) -> None:
    seen: list[list[str]] = []

    @tslab.stage(name='fit', output=tslab.Json, in_place=True)
    def fit(
        *, run: tslab.Run, epochs: int, crash: Annotated[bool, tslab.NotFingerprinted]
    ) -> tslab.Json:
        directory = run.instance_directory(stage_name='fit', instance_name=None)
        seen.append(sorted(item.name for item in directory.iterdir()))
        (directory / 'last.ckpt').write_text(str(epochs), encoding='utf-8')
        if crash:
            message = 'time limit'
            raise SimulatedTimeLimitError(message)
        return tslab.Json(data={'epochs': epochs})

    with open_run(tmp_path) as run:
        fit(run=run, epochs=1, crash=True)
    with open_run(tmp_path) as run:
        fit(run=run, epochs=1, crash=False)
    with open_run(tmp_path) as run:
        fit(run=run, epochs=2, crash=False)
    directory = tmp_path / 'r' / 'fit' / 'default'
    assert seen[0] == ['.incomplete']
    assert seen[1] == ['.incomplete', 'last.ckpt']
    assert seen[2] == ['.incomplete']
    assert sorted(item.name for item in directory.iterdir()) == [
        'data.json',
        'last.ckpt',
        'meta.json',
    ]

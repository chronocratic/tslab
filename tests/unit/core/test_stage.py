from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
from typing import Annotated, override

import pytest
import torch

import tslab
from tslab.core.status import read_status


@dataclass(frozen=True, kw_only=True)
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


def run_with_options(tmp_path: Path, **options: object) -> tslab.Run:
    return tslab.Run(
        config=Config(),
        root=tmp_path,
        relative_path=Path('r'),
        options=tslab.RunOptions(**options),  # ty: ignore[invalid-argument-type]
    )


def test_failed_rerun_keeps_the_previous_good_result(tmp_path: Path) -> None:
    fail = False

    @tslab.stage(name='flaky', output=tslab.Json)
    def flaky(*, run: tslab.Run) -> tslab.Json:
        if fail:
            message = 'transient out of memory'
            raise MemoryError(message)
        return tslab.Json(data={'ok': True})

    with open_run(tmp_path) as run:
        flaky(run=run)
    fail = True
    with run_with_options(tmp_path, rerun=('flaky[default]',)) as run:
        result = flaky(run=run)
    assert not tslab.is_available(artifact=result)
    folder = tmp_path / 'r' / 'flaky' / 'default'
    assert json.loads((folder / 'data.json').read_text(encoding='utf-8')) == {'ok': True}
    assert 'MemoryError' in (folder / 'error.txt').read_text(encoding='utf-8')
    assert not list(folder.parent.glob('*.tmp-*'))
    fail = False
    with open_run(tmp_path) as run:
        again = flaky(run=run)
    assert isinstance(again, tslab.Json)
    assert not (folder / 'error.txt').exists()


def test_rerun_from_applies_after_a_skipped_stage(tmp_path: Path) -> None:
    executed: list[str] = []

    @tslab.stage(name='first', output=tslab.Json)
    def first(*, run: tslab.Run, fail: bool) -> tslab.Json:
        if fail:
            message = 'boom'
            raise RuntimeError(message)
        return tslab.Json(data={})

    @tslab.stage(name='second', output=tslab.Json)
    def second(*, run: tslab.Run, upstream: tslab.Json) -> tslab.Json:
        executed.append('second')
        return tslab.Json(data={})

    @tslab.stage(name='third', output=tslab.Json)
    def third(*, run: tslab.Run) -> tslab.Json:
        executed.append('third')
        return tslab.Json(data={})

    with open_run(tmp_path) as run:
        second(run=run, upstream=first(run=run, fail=False))
        third(run=run)
    executed.clear()
    # second is skipped because first fails, yet it is the rerun-from stage: third must rerun.
    with run_with_options(tmp_path, rerun_from='second', force_all=False, fail_fast=False) as run:
        second(run=run, upstream=first(run=run, fail=True))
        third(run=run)
    assert executed == ['third']


def test_unwritable_work_directory_is_a_stage_failure(tmp_path: Path) -> None:
    run = open_run(tmp_path)
    with run:
        (tmp_path / 'r' / 'count').write_text('a file where the stage folder must go', 'utf-8')
        result = count(run=run, value=1)
    assert not tslab.is_available(artifact=result)
    status = json.loads((tmp_path / 'r' / 'status.json').read_text(encoding='utf-8'))
    assert status['instances']['count[default]']['state'] == 'failed'


def test_instance_meta_records_the_environment(tmp_path: Path) -> None:
    with open_run(tmp_path) as run:
        count(run=run, value=1)
    meta = json.loads(
        (tmp_path / 'r' / 'count' / 'default' / 'meta.json').read_text(encoding='utf-8')
    )
    assert set(meta['environment']) == {'git', 'versions'}
    assert meta['environment']['versions']['python']


def test_dry_run_survives_unreadable_meta(tmp_path: Path) -> None:
    with open_run(tmp_path) as run:
        count(run=run, value=1)
    meta_path = tmp_path / 'r' / 'count' / 'default' / 'meta.json'
    meta_path.write_text('{broken', encoding='utf-8')
    with run_with_options(tmp_path, dry_run=True) as run:
        placeholder = count(run=run, value=1)
    assert not tslab.is_available(artifact=placeholder)


def test_concrete_stage_class_must_define_name_and_output_type() -> None:
    with pytest.raises(TypeError, match='must define name and output_type'):

        class Nameless(tslab.Stage[tslab.Json]):
            @override
            def execute(self, **arguments: object) -> tslab.Json:
                return tslab.Json(data={})

    with pytest.raises(TypeError, match='must define output_type'):

        class Typeless(tslab.Stage[tslab.Json]):
            name = 'typeless'

            @override
            def execute(self, **arguments: object) -> tslab.Json:
                return tslab.Json(data={})


def test_var_positional_is_reported_by_name() -> None:
    with pytest.raises(TypeError, match=r"positional: \['args'\]"):
        tslab.stage(name='varargs', output=tslab.Json)(
            lambda *args, run: tslab.Json(data={})  # noqa: ARG005
        )


def test_usr2_in_a_stage_fails_it_as_signalled_and_exits(tmp_path: Path) -> None:
    assert hasattr(tslab, 'JobTimeLimitSignal')

    @tslab.stage(name='slow', output=tslab.Json)
    def slow(*, run: tslab.Run) -> tslab.Json:
        os.kill(os.getpid(), signal.SIGUSR2)
        return tslab.Json(data={})

    with pytest.raises(tslab.JobTimeLimitSignal), open_run(tmp_path) as run:
        slow(run=run)
    status = read_status(path=tmp_path / 'r' / 'status.json')
    assert status is not None
    assert status.instances['slow[default]'].state is tslab.InstanceState.FAILED
    assert (status.instances['slow[default]'].error or '').startswith('JobTimeLimitSignal:')
    assert status.run_state is tslab.RunState.CRASHED
    assert 'JobTimeLimitSignal' in (tmp_path / 'r' / 'slow' / 'default' / 'error.txt').read_text()


def test_usr2_between_stages_fails_the_next_stage_as_signalled(tmp_path: Path) -> None:
    assert hasattr(tslab, 'JobTimeLimitSignal')

    @tslab.stage(name='first', output=tslab.Json)
    def first(*, run: tslab.Run) -> tslab.Json:
        return tslab.Json(data={})

    @tslab.stage(name='second', output=tslab.Json)
    def second(*, run: tslab.Run) -> tslab.Json:
        return tslab.Json(data={})

    with (  # noqa: PT012 # why: the signal must arrive between two stage calls
        pytest.raises(tslab.JobTimeLimitSignal),
        open_run(tmp_path) as run,
    ):
        first(run=run)
        os.kill(os.getpid(), signal.SIGUSR2)
        second(run=run)
    status = read_status(path=tmp_path / 'r' / 'status.json')
    assert status is not None
    assert status.instances['first[default]'].state is tslab.InstanceState.DONE
    assert (status.instances['second[default]'].error or '').startswith('JobTimeLimitSignal:')


def test_usr2_after_the_last_stage_finishes_the_run(tmp_path: Path) -> None:
    assert hasattr(tslab, 'JobTimeLimitSignal')
    with open_run(tmp_path):
        os.kill(os.getpid(), signal.SIGUSR2)
    status = read_status(path=tmp_path / 'r' / 'status.json')
    assert status is not None
    assert status.run_state is tslab.RunState.FINISHED


def test_sigterm_after_usr2_is_ignored(tmp_path: Path) -> None:
    assert hasattr(tslab, 'JobTimeLimitSignal')

    @tslab.stage(name='slow', output=tslab.Json)
    def slow(*, run: tslab.Run) -> tslab.Json:
        try:
            os.kill(os.getpid(), signal.SIGUSR2)
        except tslab.JobTimeLimitSignal:
            os.kill(os.getpid(), signal.SIGTERM)
            raise
        return tslab.Json(data={})

    with pytest.raises(tslab.JobTimeLimitSignal), open_run(tmp_path) as run:
        slow(run=run)
    status = read_status(path=tmp_path / 'r' / 'status.json')
    assert status is not None
    assert (status.instances['slow[default]'].error or '').startswith('JobTimeLimitSignal:')

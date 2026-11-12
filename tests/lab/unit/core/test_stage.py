from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
from typing import Annotated, override

import pytest

import tslab.lab
from tslab.lab.core.status import read_status


class Opaque:
    pass


@dataclass(frozen=True, kw_only=True)
class Config:
    seed: int = 0


calls: list[str] = []


def open_run(tmp_path: Path) -> tslab.lab.Run:
    return tslab.lab.Run(
        config=Config(), root=tmp_path, relative_path=Path('r'), options=tslab.lab.RunOptions()
    )


@tslab.lab.stage(name='count', output=tslab.lab.Json)
def count(
    *, run: tslab.lab.Run, value: int, workers: Annotated[int, tslab.lab.NotFingerprinted] = 0
) -> tslab.lab.Json:
    calls.append(f'count {value} {workers}')
    return tslab.lab.Json(data={'value': value})


def test_not_fingerprinted_argument_does_not_trigger_rerun(tmp_path: Path) -> None:
    calls.clear()
    with open_run(tmp_path) as run:
        count(run=run, value=1, workers=0)
    with open_run(tmp_path) as run:
        count(run=run, value=1, workers=8)
    assert calls == ['count 1 0']


def test_version_bump_reruns(tmp_path: Path) -> None:
    @tslab.lab.stage(name='versioned', output=tslab.lab.Json, version=1)
    def first(*, run: tslab.lab.Run) -> tslab.lab.Json:
        calls.append('v1')
        return tslab.lab.Json(data={})

    @tslab.lab.stage(name='versioned', output=tslab.lab.Json, version=2)
    def second(*, run: tslab.lab.Run) -> tslab.lab.Json:
        calls.append('v2')
        return tslab.lab.Json(data={})

    calls.clear()
    with open_run(tmp_path) as run:
        first(run=run)
    with open_run(tmp_path) as run:
        second(run=run)
    assert calls == ['v1', 'v2']


def test_unfingerprintable_argument_names_the_argument(tmp_path: Path) -> None:
    @tslab.lab.stage(name='bad', output=tslab.lab.Json)
    def bad(*, run: tslab.lab.Run, model: Opaque) -> tslab.lab.Json:
        return tslab.lab.Json(data={})

    with (
        open_run(tmp_path) as run,
        pytest.raises(TypeError, match=r"bad\[default\]: cannot fingerprint argument 'model'"),
    ):
        bad(run=run, model=Opaque())


def test_wrong_return_type_is_a_stage_failure(tmp_path: Path) -> None:
    @tslab.lab.stage(name='wrong', output=tslab.lab.Metrics)
    def wrong(*, run: tslab.lab.Run) -> tslab.lab.Metrics:
        return tslab.lab.Json(data={})  # ty: ignore[invalid-return-type]

    with open_run(tmp_path) as run:
        result = wrong(run=run)
    assert not tslab.lab.is_available(artifact=result)
    status = json.loads((tmp_path / 'r' / 'status.json').read_text(encoding='utf-8'))
    assert (
        status['instances']['wrong[default]']['error']
        == "TypeError: stage 'wrong' must return Metrics, got Json"
    )


def test_decorator_requires_keyword_only_and_run() -> None:
    # why: the parameter exists only to be rejected
    with pytest.raises(TypeError, match='keyword-only'):
        tslab.lab.stage(name='positional', output=tslab.lab.Json)(
            lambda run: tslab.lab.Json(data={})  # noqa: ARG005
        )
    with pytest.raises(TypeError, match='must take a run parameter'):

        @tslab.lab.stage(name='norun', output=tslab.lab.Json)
        def no_run(*, value: int) -> tslab.lab.Json:
            return tslab.lab.Json(data={})


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
    assert isinstance(result, tslab.lab.Json)


def test_corrupt_instance_meta_is_recomputed(tmp_path: Path) -> None:
    calls.clear()
    with open_run(tmp_path) as run:
        count(run=run, value=2)
    (tmp_path / 'r' / 'count' / 'default' / 'meta.json').write_text('not json', encoding='utf-8')
    with open_run(tmp_path) as run:
        result = count(run=run, value=2)
    assert calls == ['count 2 0', 'count 2 0']
    assert isinstance(result, tslab.lab.Json)
    (tmp_path / 'r' / 'count' / 'default' / 'meta.json').write_bytes(b'\xff\xfe\x00\x01')
    with open_run(tmp_path) as run:
        result = count(run=run, value=2)
    assert calls == ['count 2 0', 'count 2 0', 'count 2 0']
    assert isinstance(result, tslab.lab.Json)


def test_external_input_is_recorded_and_content_sensitive(tmp_path: Path) -> None:
    @tslab.lab.stage(name='use', output=tslab.lab.Json)
    def use(*, run: tslab.lab.Run, thresholds: tslab.lab.Json) -> tslab.lab.Json:
        calls.append('use')
        return tslab.lab.Json(data=thresholds.data)

    source = tmp_path / 'thresholds.json'
    source.write_text('{"t": 1}', encoding='utf-8')
    calls.clear()
    for _ in range(2):
        with open_run(tmp_path) as run:
            use(run=run, thresholds=tslab.lab.Json.external(path=source))
    source.write_text('{"t": 2}', encoding='utf-8')
    with open_run(tmp_path) as run:
        use(run=run, thresholds=tslab.lab.Json.external(path=source))
    assert calls == ['use', 'use']
    meta = json.loads(
        (tmp_path / 'r' / 'use' / 'default' / 'meta.json').read_text(encoding='utf-8')
    )
    assert meta['external_inputs']['thresholds']['path'] == source.as_posix()
    assert meta['inputs'] == {}


class Doubler(tslab.lab.Stage[tslab.lab.Json]):
    name = 'double'
    output_type = tslab.lab.Json

    @override
    def execute(  # ty: ignore[invalid-method-override]
        self, *, run: tslab.lab.Run, value: int
    ) -> tslab.lab.Json:
        calls.append('double')
        return tslab.lab.Json(data={'value': 2 * value})


def test_class_based_stage(tmp_path: Path) -> None:
    calls.clear()
    for _ in range(2):
        with open_run(tmp_path) as run:
            result = Doubler()(run=run, value=3)
    assert calls == ['double']
    assert isinstance(result, tslab.lab.Json)
    assert result.data == {'value': 6}


class SimulatedTimeLimitError(RuntimeError):
    pass


def test_in_place_stage_resumes_and_restarts_on_fingerprint_change(tmp_path: Path) -> None:
    seen: list[list[str]] = []

    @tslab.lab.stage(name='fit', output=tslab.lab.Json, in_place=True)
    def fit(
        *, run: tslab.lab.Run, epochs: int, crash: Annotated[bool, tslab.lab.NotFingerprinted]
    ) -> tslab.lab.Json:
        directory = run.instance_directory(stage_name='fit', instance_name=None)
        seen.append(sorted(item.name for item in directory.iterdir()))
        (directory / 'last.ckpt').write_text(str(epochs), encoding='utf-8')
        if crash:
            message = 'time limit'
            raise SimulatedTimeLimitError(message)
        return tslab.lab.Json(data={'epochs': epochs})

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


def run_with_options(tmp_path: Path, **options: object) -> tslab.lab.Run:
    return tslab.lab.Run(
        config=Config(),
        root=tmp_path,
        relative_path=Path('r'),
        options=tslab.lab.RunOptions(**options),  # ty: ignore[invalid-argument-type]
    )


def test_failed_rerun_keeps_the_previous_good_result(tmp_path: Path) -> None:
    fail = False

    @tslab.lab.stage(name='flaky', output=tslab.lab.Json)
    def flaky(*, run: tslab.lab.Run) -> tslab.lab.Json:
        if fail:
            message = 'transient out of memory'
            raise MemoryError(message)
        return tslab.lab.Json(data={'ok': True})

    with open_run(tmp_path) as run:
        flaky(run=run)
    fail = True
    with run_with_options(tmp_path, rerun=('flaky[default]',)) as run:
        result = flaky(run=run)
    assert not tslab.lab.is_available(artifact=result)
    folder = tmp_path / 'r' / 'flaky' / 'default'
    assert json.loads((folder / 'data.json').read_text(encoding='utf-8')) == {'ok': True}
    assert 'MemoryError' in (folder / 'error.txt').read_text(encoding='utf-8')
    assert not list(folder.parent.glob('*.tmp-*'))
    fail = False
    with open_run(tmp_path) as run:
        again = flaky(run=run)
    assert isinstance(again, tslab.lab.Json)
    assert not (folder / 'error.txt').exists()


def test_rerun_from_applies_after_a_skipped_stage(tmp_path: Path) -> None:
    executed: list[str] = []

    @tslab.lab.stage(name='first', output=tslab.lab.Json)
    def first(*, run: tslab.lab.Run, fail: bool) -> tslab.lab.Json:
        if fail:
            message = 'boom'
            raise RuntimeError(message)
        return tslab.lab.Json(data={})

    @tslab.lab.stage(name='second', output=tslab.lab.Json)
    def second(*, run: tslab.lab.Run, upstream: tslab.lab.Json) -> tslab.lab.Json:
        executed.append('second')
        return tslab.lab.Json(data={})

    @tslab.lab.stage(name='third', output=tslab.lab.Json)
    def third(*, run: tslab.lab.Run) -> tslab.lab.Json:
        executed.append('third')
        return tslab.lab.Json(data={})

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
    assert not tslab.lab.is_available(artifact=result)
    status = json.loads((tmp_path / 'r' / 'status.json').read_text(encoding='utf-8'))
    assert status['instances']['count[default]']['state'] == 'failed'


def test_instance_meta_records_the_environment(tmp_path: Path) -> None:
    with open_run(tmp_path) as run:
        count(run=run, value=1)
    meta = json.loads(
        (tmp_path / 'r' / 'count' / 'default' / 'meta.json').read_text(encoding='utf-8')
    )
    assert set(meta['environment']) == {'git', 'hardware', 'versions'}
    assert meta['environment']['versions']['python']


def test_dry_run_survives_unreadable_meta(tmp_path: Path) -> None:
    with open_run(tmp_path) as run:
        count(run=run, value=1)
    meta_path = tmp_path / 'r' / 'count' / 'default' / 'meta.json'
    meta_path.write_text('{broken', encoding='utf-8')
    with run_with_options(tmp_path, dry_run=True) as run:
        placeholder = count(run=run, value=1)
    assert not tslab.lab.is_available(artifact=placeholder)


def test_concrete_stage_class_must_define_name_and_output_type() -> None:
    with pytest.raises(TypeError, match='must define name and output_type'):

        class Nameless(tslab.lab.Stage[tslab.lab.Json]):
            @override
            def execute(self, **arguments: object) -> tslab.lab.Json:
                return tslab.lab.Json(data={})

    with pytest.raises(TypeError, match='must define output_type'):

        class Typeless(tslab.lab.Stage[tslab.lab.Json]):
            name = 'typeless'

            @override
            def execute(self, **arguments: object) -> tslab.lab.Json:
                return tslab.lab.Json(data={})


def test_var_positional_is_reported_by_name() -> None:
    with pytest.raises(TypeError, match=r"positional: \['args'\]"):
        tslab.lab.stage(name='varargs', output=tslab.lab.Json)(
            lambda *args, run: tslab.lab.Json(data={})  # noqa: ARG005
        )


def test_usr2_in_a_stage_fails_it_as_signaled_and_exits(tmp_path: Path) -> None:
    assert hasattr(tslab.lab, 'JobTimeLimitSignal')

    @tslab.lab.stage(name='slow', output=tslab.lab.Json)
    def slow(*, run: tslab.lab.Run) -> tslab.lab.Json:
        os.kill(os.getpid(), signal.SIGUSR2)
        return tslab.lab.Json(data={})

    with pytest.raises(tslab.lab.JobTimeLimitSignal), open_run(tmp_path) as run:
        slow(run=run)
    status = read_status(path=tmp_path / 'r' / 'status.json')
    assert status is not None
    assert status.instances['slow[default]'].state is tslab.lab.InstanceState.FAILED
    assert (status.instances['slow[default]'].error or '').startswith('JobTimeLimitSignal:')
    assert status.run_state is tslab.lab.RunState.CRASHED
    assert 'JobTimeLimitSignal' in (tmp_path / 'r' / 'slow' / 'default' / 'error.txt').read_text()


def test_usr2_between_stages_fails_the_next_stage_as_signaled(tmp_path: Path) -> None:
    assert hasattr(tslab.lab, 'JobTimeLimitSignal')

    @tslab.lab.stage(name='first', output=tslab.lab.Json)
    def first(*, run: tslab.lab.Run) -> tslab.lab.Json:
        return tslab.lab.Json(data={})

    @tslab.lab.stage(name='second', output=tslab.lab.Json)
    def second(*, run: tslab.lab.Run) -> tslab.lab.Json:
        return tslab.lab.Json(data={})

    with (  # noqa: PT012 # why: the signal must arrive between two stage calls
        pytest.raises(tslab.lab.JobTimeLimitSignal),
        open_run(tmp_path) as run,
    ):
        first(run=run)
        os.kill(os.getpid(), signal.SIGUSR2)
        second(run=run)
    status = read_status(path=tmp_path / 'r' / 'status.json')
    assert status is not None
    assert status.instances['first[default]'].state is tslab.lab.InstanceState.DONE
    assert (status.instances['second[default]'].error or '').startswith('JobTimeLimitSignal:')


def test_signal_before_preparation_does_not_resume_checkpoint_from_another_fingerprint(
    tmp_path: Path,
) -> None:
    observed: list[tuple[int, bool]] = []

    @tslab.lab.stage(name='fit', output=tslab.lab.Json, in_place=True)
    def fit(*, run: tslab.lab.Run, settings: int, crash: bool) -> tslab.lab.Json:
        directory = run.instance_directory(stage_name='fit', instance_name=None)
        observed.append((settings, (directory / 'last.ckpt').exists()))
        (directory / 'last.ckpt').write_text(str(settings), encoding='utf-8')
        if crash:
            message = 'interrupted'
            raise RuntimeError(message)
        return tslab.lab.Json(data={'settings': settings})

    with open_run(tmp_path) as run:
        fit(run=run, settings=1, crash=True)
    with open_run(tmp_path) as run:
        os.kill(os.getpid(), signal.SIGUSR2)
        with pytest.raises(tslab.lab.JobTimeLimitSignal):
            fit(run=run, settings=2, crash=False)
    with open_run(tmp_path) as run:
        fit(run=run, settings=2, crash=False)

    assert observed == [(1, False), (2, False)]


def test_signal_before_forced_preparation_does_not_resume_old_checkpoint(tmp_path: Path) -> None:
    observed: list[bool] = []

    @tslab.lab.stage(name='fit', output=tslab.lab.Json, in_place=True)
    def fit(*, run: tslab.lab.Run) -> tslab.lab.Json:
        directory = run.instance_directory(stage_name='fit', instance_name=None)
        observed.append((directory / 'last.ckpt').exists())
        (directory / 'last.ckpt').write_text('checkpoint', encoding='utf-8')
        return tslab.lab.Json(data={})

    with open_run(tmp_path) as run:
        fit(run=run)
    with run_with_options(tmp_path, force_all=True) as run:
        os.kill(os.getpid(), signal.SIGUSR2)
        with pytest.raises(tslab.lab.JobTimeLimitSignal):
            fit(run=run)
    with open_run(tmp_path) as run:
        fit(run=run)

    assert observed == [False, False]


def test_usr2_after_the_last_stage_finishes_the_run(tmp_path: Path) -> None:
    assert hasattr(tslab.lab, 'JobTimeLimitSignal')
    with open_run(tmp_path):
        os.kill(os.getpid(), signal.SIGUSR2)
    status = read_status(path=tmp_path / 'r' / 'status.json')
    assert status is not None
    assert status.run_state is tslab.lab.RunState.FINISHED


def test_sigterm_after_usr2_is_ignored(tmp_path: Path) -> None:
    assert hasattr(tslab.lab, 'JobTimeLimitSignal')

    @tslab.lab.stage(name='slow', output=tslab.lab.Json)
    def slow(*, run: tslab.lab.Run) -> tslab.lab.Json:
        try:
            os.kill(os.getpid(), signal.SIGUSR2)
        except tslab.lab.JobTimeLimitSignal:
            os.kill(os.getpid(), signal.SIGTERM)
            raise
        return tslab.lab.Json(data={})

    with pytest.raises(tslab.lab.JobTimeLimitSignal), open_run(tmp_path) as run:
        slow(run=run)
    status = read_status(path=tmp_path / 'r' / 'status.json')
    assert status is not None
    assert (status.instances['slow[default]'].error or '').startswith('JobTimeLimitSignal:')


def test_non_zero_rank_leaves_at_a_single_rank_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('SLURM_PROCID', '1')
    with pytest.raises(SystemExit) as raised, open_run(tmp_path) as run:
        count(run=run, value=2)
    assert raised.value.code == 0
    assert not (tmp_path / 'r').exists()


def test_non_zero_rank_runs_an_all_ranks_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('SLURM_PROCID', '1')

    @tslab.lab.stage(name='everywhere', output=tslab.lab.Json, all_ranks=True)
    def everywhere(*, run: tslab.lab.Run) -> tslab.lab.Json:
        return tslab.lab.Json(data={'executed': True})

    with open_run(tmp_path) as run:
        result = everywhere(run=run)
    assert result.data == {'executed': True}
    assert not (tmp_path / 'r').exists()


class FakeAgreement:
    def __init__(self, *, cache: bool | None = None) -> None:
        self.cache = cache
        self.keys: list[str] = []

    def agree_across_ranks(self, *, key: str, value: bool) -> bool:
        self.keys.append(key)
        return self.cache if key.endswith(':cache') and self.cache is not None else value


def test_all_ranks_stage_after_a_single_rank_stage_fails_in_a_multi_rank_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('SLURM_NTASKS', '2')

    @tslab.lab.stage(name='everywhere', output=tslab.lab.Json, all_ranks=True)
    def everywhere(
        *, run: tslab.lab.Run, agreement: Annotated[FakeAgreement, tslab.lab.NotFingerprinted]
    ) -> tslab.lab.Json:
        return tslab.lab.Json(data={})

    with open_run(tmp_path) as run:
        count(run=run, value=2)
        with pytest.raises(RuntimeError, match='must come before'):
            everywhere(run=run, agreement=FakeAgreement())


def test_all_ranks_stage_follows_the_agreed_cache_decision(tmp_path: Path) -> None:
    executed: list[int] = []

    @tslab.lab.stage(name='everywhere', output=tslab.lab.Json, all_ranks=True)
    def everywhere(
        *, run: tslab.lab.Run, agreement: Annotated[FakeAgreement, tslab.lab.NotFingerprinted]
    ) -> tslab.lab.Json:
        executed.append(1)
        return tslab.lab.Json(data={'value': len(executed)})

    first_agreement = FakeAgreement()
    with open_run(tmp_path) as run:
        everywhere(run=run, agreement=first_agreement)
    agreement = FakeAgreement(cache=False)
    with open_run(tmp_path) as run:
        result = everywhere(run=run, agreement=agreement)
    assert result.data == {'value': 2}
    assert agreement.keys != first_agreement.keys
    assert [key.rsplit(':', maxsplit=1)[-1] for key in agreement.keys] == ['cache', 'prepared']

from collections.abc import Callable
import dataclasses
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

import pytest

from tests.lab.integration.toy_pipeline import main, make_data, read_calls, summarize, ToyConfig
import tslab.lab


def toy(tmp_path: Path, **changes: object) -> ToyConfig:
    return dataclasses.replace(ToyConfig(root=tmp_path, relative_path=Path('toy')), **changes)


def status_of(tmp_path: Path) -> dict[str, Any]:
    return json.loads((tmp_path / 'toy' / 'status.json').read_text(encoding='utf-8'))


def test_first_run_executes_everything_and_records_status(tmp_path: Path) -> None:
    results = main(config=toy(tmp_path))
    assert read_calls(root=tmp_path) == [
        'make_data',
        'transform[x1]',
        'summarize[x1]',
        'transform[x2]',
        'summarize[x2]',
    ]
    status = status_of(tmp_path)
    assert status['run_state'] == 'finished'
    assert {record['state'] for record in status['instances'].values()} == {'done'}
    metrics = results['x2']
    assert isinstance(metrics, tslab.lab.Metrics)
    assert (tmp_path / 'toy' / 'transform' / 'x2' / 'tensors.safetensors').exists()
    assert (tmp_path / 'toy' / 'config.yaml').exists()
    assert (tmp_path / 'toy' / 'logs' / 'run.log').exists()
    assert not (tmp_path / 'toy' / '.lock').exists()


def test_second_run_loads_everything(tmp_path: Path) -> None:
    first = main(config=toy(tmp_path))
    second = main(config=toy(tmp_path))
    assert len(read_calls(root=tmp_path)) == 5
    assert second['x1'].values == first['x1'].values  # ty: ignore[unresolved-attribute]


def test_changed_setting_reruns_only_affected_instances(tmp_path: Path) -> None:
    main(config=toy(tmp_path))
    main(config=toy(tmp_path, scales=(1.0, 3.0)))
    assert read_calls(root=tmp_path)[5:] == ['transform[x3]', 'summarize[x3]']


def test_changed_upstream_reruns_downstream(tmp_path: Path) -> None:
    main(config=toy(tmp_path))
    main(config=toy(tmp_path, size=5))
    assert read_calls(root=tmp_path)[5:] == [
        'make_data',
        'transform[x1]',
        'summarize[x1]',
        'transform[x2]',
        'summarize[x2]',
    ]


def test_seed_is_part_of_fingerprint(tmp_path: Path) -> None:
    main(config=toy(tmp_path))
    main(config=toy(tmp_path, seed=1))
    assert len(read_calls(root=tmp_path)) == 10


def test_failure_skips_dependants_and_rerun_repeats_only_them(tmp_path: Path) -> None:
    main(config=toy(tmp_path, fail_scale=2.0))
    status = status_of(tmp_path)
    assert status['run_state'] == 'finished'
    assert status['instances']['transform[x2]']['state'] == 'failed'
    assert (
        status['instances']['transform[x2]']['error'] == 'RuntimeError: scale 2.0 fails on purpose'
    )
    assert (tmp_path / 'toy' / 'transform' / 'x2' / 'error.txt').exists()
    assert status['instances']['summarize[x2]'] == {
        'state': 'skipped',
        'reason': 'input transform[x2] failed',
    }
    assert status['instances']['summarize[x1]']['state'] == 'done'
    main(config=toy(tmp_path))
    assert read_calls(root=tmp_path)[4:] == ['transform[x2]', 'summarize[x2]']


def test_fail_fast_reraises_and_marks_run_failed(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match='fails on purpose'):
        main(config=toy(tmp_path, fail_scale=1.0, run=tslab.lab.RunOptions(fail_fast=True)))
    assert status_of(tmp_path)['run_state'] == 'failed'
    assert not (tmp_path / 'toy' / '.lock').exists()


def test_force_all_reruns_everything(tmp_path: Path) -> None:
    main(config=toy(tmp_path))
    main(config=toy(tmp_path, run=tslab.lab.RunOptions(force_all=True)))
    assert len(read_calls(root=tmp_path)) == 10


def test_rerun_one_instance_also_reruns_its_dependants(tmp_path: Path) -> None:
    main(config=toy(tmp_path))
    main(config=toy(tmp_path, run=tslab.lab.RunOptions(rerun=('transform[x1]',))))
    assert read_calls(root=tmp_path)[5:] == ['transform[x1]', 'summarize[x1]']


def test_rerun_from_reruns_stage_and_everything_after(tmp_path: Path) -> None:
    main(config=toy(tmp_path))
    main(config=toy(tmp_path, run=tslab.lab.RunOptions(rerun_from='transform')))
    assert read_calls(root=tmp_path)[5:] == [
        'transform[x1]',
        'summarize[x1]',
        'transform[x2]',
        'summarize[x2]',
    ]


def test_dry_run_executes_and_writes_nothing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    main(config=toy(tmp_path))
    before = (tmp_path / 'toy' / 'status.json').read_text(encoding='utf-8')
    caplog.set_level('INFO')
    results = main(config=toy(tmp_path, scales=(1.0, 3.0), run=tslab.lab.RunOptions(dry_run=True)))
    assert len(read_calls(root=tmp_path)) == 5
    assert (tmp_path / 'toy' / 'status.json').read_text(encoding='utf-8') == before
    assert not tslab.lab.is_available(artifact=results['x1'])
    assert 'make_data[default] -> load' in caplog.text
    assert 'summarize[x1] -> load' in caplog.text
    assert 'transform[x3] -> run (new)' in caplog.text
    assert 'summarize[x3] -> run (upstream changes)' in caplog.text


def test_duplicate_unnamed_call_raises(tmp_path: Path) -> None:
    with tslab.lab.Run(config=toy(tmp_path), root=tmp_path, relative_path=Path('dup')) as run:
        make_data(run=run, size=2)
        with pytest.raises(ValueError, match='called twice without name='):
            make_data(run=run, size=3)


def test_nested_name_conflict_raises(tmp_path: Path) -> None:
    with tslab.lab.Run(config=toy(tmp_path), root=tmp_path, relative_path=Path('nested')) as run:
        data = make_data(run=run, size=2)
        summarize(run=run, name=('a', 'b'), data=data)
        with pytest.raises(ValueError, match='conflicts'):
            summarize(run=run, name='a', data=data)


def test_name_on_single_instance_stage_raises(tmp_path: Path) -> None:
    with (
        tslab.lab.Run(config=toy(tmp_path), root=tmp_path, relative_path=Path('single')) as run,
        pytest.raises(TypeError, match='single-instance'),
    ):
        make_data(run=run, name='x', size=2)  # ty: ignore[unknown-argument]


def wait_for(predicate: Callable[[], bool], timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.1)
    message = 'condition not reached in time'
    raise AssertionError(message)


@pytest.mark.slow
def test_killed_run_resumes_where_it_stopped(tmp_path: Path) -> None:
    script = Path(__file__).with_name('toy_pipeline.py')
    environment = {**os.environ, 'TOY_SLEEP_IN': 'transform[x2]'}
    process = subprocess.Popen(  # noqa: S603 # why: runs this repository's own script
        [sys.executable, str(script), '--root', str(tmp_path), '--relative-path', 'toy'],
        env=environment,
    )
    wait_for(lambda: 'transform[x2]' in read_calls(root=tmp_path))
    process.send_signal(signal.SIGKILL)
    process.wait()
    assert status_of(tmp_path)['instances']['transform[x2]']['state'] == 'running'
    assert (tmp_path / 'toy' / '.lock').exists()
    main(config=toy(tmp_path))
    assert read_calls(root=tmp_path)[4:] == ['transform[x2]', 'summarize[x2]']
    assert status_of(tmp_path)['run_state'] == 'finished'
    assert not list((tmp_path / 'toy' / 'transform').glob('*.tmp-*'))

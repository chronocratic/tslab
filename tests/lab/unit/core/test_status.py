import json
from pathlib import Path

import pytest

from tslab.lab.core.status import (
    format_instance_id,
    InstanceRecord,
    parse_instance_id,
    read_status,
    RunStatusSnapshot,
    shown_state_of,
    StatusFileError,
    write_status,
)
from tslab.lab.enums import InstanceState, RunState


def test_round_trip(tmp_path: Path) -> None:
    snapshot = RunStatusSnapshot(run_state=RunState.RUNNING, updated_at='t').with_instance(
        instance_id='scale[sma]',
        record=InstanceRecord(state=InstanceState.DONE, fingerprint='f' * 16),
    )
    write_status(path=tmp_path / 'status.json', snapshot=snapshot)
    loaded = read_status(path=tmp_path / 'status.json')
    assert loaded is not None
    assert loaded.instances == snapshot.instances
    document = json.loads((tmp_path / 'status.json').read_text(encoding='utf-8'))
    assert document['instances']['scale[sma]'] == {'state': 'done', 'fingerprint': 'f' * 16}


def test_missing_file_reads_as_none(tmp_path: Path) -> None:
    assert read_status(path=tmp_path / 'status.json') is None


def test_instance_id_format() -> None:
    assert format_instance_id(stage_name='scale', name_parts=('forecasting', 'h24', 'sma')) == (
        'scale[forecasting/h24/sma]'
    )


def test_parse_instance_id_inverts_format_instance_id() -> None:
    instance_id = format_instance_id(stage_name='scale', name_parts=('forecasting', 'h24', 'sma'))
    assert parse_instance_id(instance_id=instance_id) == ('scale', ('forecasting', 'h24', 'sma'))
    assert parse_instance_id(instance_id='encode[default]') == ('encode', ('default',))


@pytest.mark.parametrize('instance_id', ['encode', 'encode[', '[x]', 'encode[x'])
def test_parse_instance_id_rejects_other_shapes(instance_id: str) -> None:
    with pytest.raises(ValueError, match='invalid instance id'):
        parse_instance_id(instance_id=instance_id)


def _document(*, instances: str) -> str:
    return f'{{"run_state": "running", "updated_at": "t", "instances": {instances}}}'


@pytest.mark.parametrize(
    'content',
    [
        '[1]',
        '{broken',
        '{}',
        _document(instances='{}').replace('running', 'nope'),
        _document(instances='5'),
        _document(instances='{"a[b]": {"state": "bogus"}}'),
        _document(instances='{"a[b]": {}}'),
        _document(instances='{"a[b]": 3}'),
    ],
)
def test_every_bad_status_file_is_one_error_naming_the_file(tmp_path: Path, content: str) -> None:
    (tmp_path / 'status.json').write_text(content, encoding='utf-8')
    with pytest.raises(StatusFileError, match=r'status\.json'):
        read_status(path=tmp_path / 'status.json')


def test_shown_state_shows_interrupted_for_a_dead_running_run() -> None:
    assert shown_state_of(run_state=RunState.RUNNING, lock_live=False) == 'interrupted'


@pytest.mark.parametrize('run_state', list(RunState))
def test_shown_state_shows_running_for_a_live_lock_in_any_saved_state(run_state: RunState) -> None:
    assert shown_state_of(run_state=run_state, lock_live=True) == 'running'


@pytest.mark.parametrize('run_state', [RunState.FINISHED, RunState.FAILED, RunState.CRASHED])
def test_shown_state_keeps_the_saved_state_without_a_live_lock(run_state: RunState) -> None:
    assert shown_state_of(run_state=run_state, lock_live=False) == run_state.value


def test_traceback_file_is_saved_as_posix_text_and_read_back_as_a_path() -> None:
    record = InstanceRecord(state=InstanceState.FAILED, traceback_file=Path('a') / 'error.txt')
    document = record.to_json()
    assert document['traceback_file'] == 'a/error.txt'
    assert InstanceRecord.from_json(document=document).traceback_file == Path('a/error.txt')


def test_traceback_file_must_be_a_string_in_the_document() -> None:
    with pytest.raises(ValueError, match='traceback_file must be a string'):
        InstanceRecord.from_json(document={'state': 'failed', 'traceback_file': 3})

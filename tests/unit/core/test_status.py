import json
from pathlib import Path

import pytest

from tslab.core.status import (
    format_instance_id,
    InstanceRecord,
    InstanceState,
    parse_instance_id,
    read_status,
    RunState,
    RunStatusSnapshot,
    StatusFileError,
    write_status,
)


def test_round_trip(tmp_path: Path) -> None:
    snapshot = RunStatusSnapshot(run_state=RunState.RUNNING, updated_at='t').with_instance(
        instance_id='transform[smooth]',
        record=InstanceRecord(state=InstanceState.DONE, fingerprint='f' * 16),
    )
    write_status(path=tmp_path / 'status.json', snapshot=snapshot)
    loaded = read_status(path=tmp_path / 'status.json')
    assert loaded is not None
    assert loaded.instances == snapshot.instances
    document = json.loads((tmp_path / 'status.json').read_text(encoding='utf-8'))
    assert document['format_version'] == 1
    assert document['instances']['transform[smooth]'] == {'state': 'done', 'fingerprint': 'f' * 16}


def test_missing_file_reads_as_none(tmp_path: Path) -> None:
    assert read_status(path=tmp_path / 'status.json') is None


def test_unknown_format_version_raises(tmp_path: Path) -> None:
    (tmp_path / 'status.json').write_text('{"format_version": 99}', encoding='utf-8')
    with pytest.raises(ValueError, match='format_version'):
        read_status(path=tmp_path / 'status.json')


def test_instance_id_format() -> None:
    assert format_instance_id(stage_name='transform', name_parts=('forecasting', 'h24', 'smooth')) == (
        'transform[forecasting/h24/smooth]'
    )


def test_parse_instance_id_inverts_format_instance_id() -> None:
    instance_id = format_instance_id(stage_name='transform', name_parts=('forecasting', 'h24', 'smooth'))
    assert parse_instance_id(instance_id=instance_id) == ('transform', ('forecasting', 'h24', 'smooth'))
    assert parse_instance_id(instance_id='encode[default]') == ('encode', ('default',))


@pytest.mark.parametrize('instance_id', ['encode', 'encode[', '[x]', 'encode[x'])
def test_parse_instance_id_rejects_other_shapes(instance_id: str) -> None:
    with pytest.raises(ValueError, match='invalid instance id'):
        parse_instance_id(instance_id=instance_id)


def _document(*, instances: str) -> str:
    return (
        '{"format_version": 1, "run_state": "running", "updated_at": "t", '
        f'"instances": {instances}}}'
    )


@pytest.mark.parametrize(
    'content',
    [
        '[1]',
        '{broken',
        '{"format_version": 1}',
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


def test_status_file_error_is_a_value_error() -> None:
    assert issubclass(StatusFileError, ValueError)

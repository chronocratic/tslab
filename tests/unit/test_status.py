import json
from pathlib import Path

import pytest

from tslab.core.status import (
    format_instance_id,
    InstanceRecord,
    InstanceState,
    read_status,
    RunState,
    RunStatusSnapshot,
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

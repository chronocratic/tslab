import logging

import pytest

from tslab.lab.core.text import (
    describe_error,
    first_line,
    format_bytes,
    format_duration,
    format_index_ranges,
    format_table,
    log_table,
    TextTable,
    unknown_key_message,
)


def test_describe_error() -> None:
    assert describe_error(error=ValueError('bad\nvalue')) == 'ValueError: bad\nvalue'


def test_first_line() -> None:
    assert first_line(text='a\nb') == 'a'
    assert first_line(text='') == ''
    assert first_line(text='\nb') == ''


def test_format_duration() -> None:
    assert format_duration(seconds=0) == '0:00:00'
    assert format_duration(seconds=59) == '0:00:59'
    assert format_duration(seconds=60) == '0:01:00'
    assert format_duration(seconds=3600) == '1:00:00'
    assert format_duration(seconds=3661) == '1:01:01'
    assert format_duration(seconds=90.7) == '0:01:30'


def test_format_bytes() -> None:
    assert format_bytes(size=1024**3) == '1.0 GiB'
    assert format_bytes(size=3 * 1024**3 // 2) == '1.5 GiB'
    assert format_bytes(size=1024**2) == '1.0 MiB'
    assert format_bytes(size=3 * 1024**2 // 2) == '1.5 MiB'
    assert format_bytes(size=1024) == '1.0 KiB'
    assert format_bytes(size=512) == '0.5 KiB'


def test_format_index_ranges() -> None:
    assert format_index_ranges(indices=(10, 9, 0, 1, 2, 3, 7, 3)) == '0-3,7,9-10'
    assert format_index_ranges(indices=(4,)) == '4'
    with pytest.raises(ValueError, match='indices'):
        format_index_ranges(indices=())


def test_format_table_aligns_left_and_strips_trailing_spaces() -> None:
    lines = format_table(header=('a', 'bb'), rows=[('long', 'x')])
    assert lines == ['a     bb', 'long  x']


def test_log_table_logs_one_plain_record_with_the_table(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level('INFO'):
        log_table(logger=logging.getLogger('t'), header=('a', 'bb'), rows=[('long', 'x')])
    (record,) = caplog.records
    assert record.getMessage() == 'a     bb\nlong  x'
    table = getattr(record, 'tslab_table', None)
    assert isinstance(table, TextTable)
    assert table.rows == (('long', 'x'),)


def test_unknown_key_message_suggests_the_close_key() -> None:
    text = unknown_key_message(location='slurm', key='gpu', valid=['gpus', 'time', 'partition'])
    assert text == "slurm.gpu: unknown key; did you mean 'gpus'? Valid keys: gpus, partition, time"


def test_unknown_key_message_without_a_close_key() -> None:
    text = unknown_key_message(location='', key='zzz', valid=['gpus', 'time'])
    assert text == 'zzz: unknown key. Valid keys: gpus, time'

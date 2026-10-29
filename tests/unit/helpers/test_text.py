import logging

import pytest

from tslab.helpers.text import first_line, format_table, log_table, TextTable, unknown_key_message


def test_first_line() -> None:
    assert first_line(text='a\nb') == 'a'
    assert first_line(text='') == ''
    assert first_line(text='\nb') == ''


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

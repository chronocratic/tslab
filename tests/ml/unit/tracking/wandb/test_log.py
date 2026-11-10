from pathlib import Path
import struct
import zlib

import pytest
from wandb.proto import wandb_internal_pb2

from tslab.tracking.wandb.log import read_exit_codes, read_records

HEADER = struct.pack('<4sHB', b':W&B', 0xBEE1, 0)
FULL = 1


def exit_record(*, code: int) -> bytes:
    # why: protobuf class is invisible to type checkers
    record = getattr(wandb_internal_pb2, 'Record')()  # noqa: B009
    record.exit.exit_code = code
    payload = bytes(record.SerializeToString())
    checksum = zlib.crc32(bytes([FULL]) + payload)
    header = bytes(struct.pack('<IHB', checksum, len(payload), FULL))
    return header + payload


def test_exit_codes_are_read_in_order(tmp_path: Path) -> None:
    path = tmp_path / 'run.wandb'
    path.write_bytes(HEADER + exit_record(code=0) + exit_record(code=130))
    assert read_exit_codes(path=path) == [0, 130]


def test_a_truncated_last_record_keeps_the_records_before_it(tmp_path: Path) -> None:
    path = tmp_path / 'run.wandb'
    path.write_bytes(HEADER + exit_record(code=0) + exit_record(code=130)[:-1])
    assert read_exit_codes(path=path) == [0]
    assert len(list(read_records(path=path))) == 1


def test_a_corrupt_record_in_the_middle_still_raises(tmp_path: Path) -> None:
    path = tmp_path / 'run.wandb'
    damaged = bytearray(exit_record(code=0))
    damaged[-1] ^= 0xFF
    path.write_bytes(HEADER + bytes(damaged) + exit_record(code=1))
    with pytest.raises(ValueError, match='corrupt'):
        read_exit_codes(path=path)


def test_a_file_that_is_not_a_wandb_log_raises(tmp_path: Path) -> None:
    path = tmp_path / 'run.wandb'
    path.write_bytes(b'nothing')
    with pytest.raises(ValueError, match='not a W&B'):
        read_exit_codes(path=path)

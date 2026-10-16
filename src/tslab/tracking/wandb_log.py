"""Reader for W&B's local transaction log (the ``run-<id>.wandb`` file).

``wandb sync`` replays this file to the server, so its exit records decide
which state a resync can reach. wandb 0.30 removed the Python reader
(``wandb.sdk.internal.datastore``), which silently turns an exit-code check
built on it into a no-op. This module reads the format directly.

Format (a LevelDB-style log, unchanged since wandb's first datastore):

- a 7-byte header: ``b':W&B'``, magic ``0xBEE1`` (little-endian), version ``0``;
- then 32 KiB blocks of records, each with a 7-byte header (CRC-32 of the
  type byte plus the payload, payload length, type) and the payload;
- type FULL holds a whole protobuf ``Record``; FIRST, MIDDLE and LAST hold the
  pieces of a record split across blocks.
"""

from collections.abc import Iterator
from pathlib import Path
import struct
from typing import Any
import zlib

__all__ = ['read_exit_codes', 'read_records', 'read_run_coordinates']

_HEADER = struct.Struct('<4sHB')
_HEADER_IDENT = b':W&B'
_HEADER_MAGIC = 0xBEE1
_RECORD_HEADER = struct.Struct('<IHB')
_BLOCK_SIZE = 32 * 1024
_FULL, _FIRST, _MIDDLE, _LAST = 1, 2, 3, 4


def _check_header(*, path: Path, data: bytes) -> None:
    """Raise ``ValueError`` unless ``data`` starts with a W&B log header."""
    if len(data) < _HEADER.size:
        ident, magic = b'', 0
    else:
        ident, magic, _version = _HEADER.unpack_from(data, 0)
    if ident != _HEADER_IDENT or magic != _HEADER_MAGIC:
        message = f'{path.as_posix()} is not a W&B transaction log'
        raise ValueError(message)


def read_records(*, path: Path) -> Iterator[bytes]:
    """Yield the raw protobuf payload of every record in a ``.wandb`` file.

    Args:
        path: A ``run-<id>.wandb`` transaction log.

    Yields:
        Serialised ``wandb_internal_pb2.Record`` messages, in log order.

    Raises:
        ValueError: If the header is not a W&B log or a record checksum fails
            (a truncated or corrupt file).
    """
    data = path.read_bytes()
    _check_header(path=path, data=data)
    offset = _HEADER.size
    pieces: list[bytes] = []
    while offset + _RECORD_HEADER.size <= len(data):
        remaining_in_block = _BLOCK_SIZE - offset % _BLOCK_SIZE
        if remaining_in_block < _RECORD_HEADER.size:
            offset += remaining_in_block  # block trailer padding
            continue
        checksum, length, kind = _RECORD_HEADER.unpack_from(data, offset)
        offset += _RECORD_HEADER.size
        if kind == 0 and length == 0:
            break  # zero padding: end of written data
        payload = data[offset : offset + length]
        offset += length
        if zlib.crc32(bytes([kind]) + payload) != checksum:
            message = f'{path.as_posix()}: corrupt record at byte {offset - length}'
            raise ValueError(message)
        if kind == _FULL:
            yield payload
        elif kind == _FIRST:
            pieces = [payload]
        elif kind == _MIDDLE:
            pieces.append(payload)
        elif kind == _LAST:
            pieces.append(payload)
            yield b''.join(pieces)
            pieces = []


def _parsed_records(*, path: Path) -> Iterator[Any]:
    """Yield every record of the log as a parsed protobuf ``Record``."""
    from wandb.proto import wandb_internal_pb2  # noqa: PLC0415 # why: optional extra

    # protobuf creates message classes at import time, so type checkers cannot see them.
    record_class = getattr(wandb_internal_pb2, 'Record')  # noqa: B009 # why: see comment above
    for payload in read_records(path=path):
        record = record_class()
        record.ParseFromString(payload)
        yield record


def read_exit_codes(*, path: Path) -> list[int]:
    """Return every exit code recorded in a ``.wandb`` transaction log, in order.

    A log can hold several exit records, for example a clean ``0`` followed by
    ``130`` when SLURM interrupts the process during teardown. The server
    applies them in order, so the last one decides the final state.

    Args:
        path: A ``run-<id>.wandb`` transaction log.

    Returns:
        The exit codes in log order (empty if the run never finished).
    """
    return [
        record.exit.exit_code
        for record in _parsed_records(path=path)
        if record.WhichOneof('record_type') == 'exit'
    ]


def read_run_coordinates(*, path: Path) -> tuple[str | None, str | None, str | None]:
    """Return ``(entity, project, run_id)`` from the log's first run record.

    wandb writes the run record at start-up, so it is present even when the run
    crashed before its debug logs showed a W&B URL. Offline runs started
    without an explicit entity record an empty entity.

    Args:
        path: A ``run-<id>.wandb`` transaction log.

    Returns:
        The coordinates; an empty or missing value is ``None``.
    """
    for record in _parsed_records(path=path):
        if record.WhichOneof('record_type') == 'run':
            run = record.run
            # protobuf string fields are '' when unset.
            return str(run.entity) or None, str(run.project) or None, str(run.run_id) or None
    return None, None, None

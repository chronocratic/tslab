"""The ``status.json`` file: run state plus one record per stage instance.

``status.json`` is a public, versioned format (``format_version``). A future
study layer reads it to decide which runs are complete, so its shape must only
change together with ``STATUS_FORMAT_VERSION``.
"""

from collections.abc import Mapping
import dataclasses
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import StrEnum
from pathlib import Path

from tslab.core.paths import (
    atomic_write_json,
    JsonFileError,
    read_json_object,
    require_path,
    STATUS_FILE,
)

__all__ = [
    'STATUS_FILE',
    'STATUS_FORMAT_VERSION',
    'InstanceRecord',
    'InstanceState',
    'RunState',
    'RunStatusSnapshot',
    'StatusFileError',
    'format_instance_id',
    'parse_instance_id',
    'read_status',
    'utc_now_iso',
    'write_status',
]

STATUS_FORMAT_VERSION = 1


class StatusFileError(JsonFileError):
    """Raised when ``status.json`` is unreadable, malformed or of another format version."""


class InstanceState(StrEnum):
    """Lifecycle of one stage instance."""

    RUNNING = 'running'
    DONE = 'done'
    FAILED = 'failed'
    SKIPPED = 'skipped'


class RunState(StrEnum):
    """Lifecycle of a whole run.

    ``FINISHED`` means the ``with Run`` block ended without an exception. It does
    not mean every instance is ``done``: failed instances are allowed.
    """

    RUNNING = 'running'
    FINISHED = 'finished'
    FAILED = 'failed'
    CRASHED = 'crashed'


@dataclass(frozen=True, kw_only=True)
class InstanceRecord:
    """What ``status.json`` stores about one stage instance.

    Attributes:
        state: Current lifecycle state.
        fingerprint: Fingerprint of the settings and inputs, or ``None`` when the
            instance was skipped before a fingerprint could be computed.
        started_at: UTC ISO 8601 time the execution started.
        finished_at: UTC ISO 8601 time the execution ended.
        seconds: Wall-clock duration of the execution.
        error: ``"ExceptionType: message"`` for failed instances.
        traceback_file: Path of ``error.txt``, relative to the run folder.
        reason: Why the instance was skipped.
    """

    state: InstanceState
    fingerprint: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    seconds: float | None = None
    error: str | None = None
    traceback_file: str | None = None
    reason: str | None = None

    def to_json(self) -> dict[str, object]:
        """Return the JSON form, omitting fields that are ``None``."""
        document: dict[str, object] = {'state': self.state.value}
        for record_field in dataclasses.fields(self):
            value = getattr(self, record_field.name)
            if record_field.name != 'state' and value is not None:
                document[record_field.name] = value
        return document

    @classmethod
    def from_json(cls, *, document: Mapping[str, object]) -> 'InstanceRecord':
        """Build a record from its JSON form.

        Args:
            document: One value of the ``instances`` mapping in ``status.json``.

        Returns:
            The parsed record.

        Raises:
            ValueError: If a field has the wrong type.
        """
        seconds = document.get('seconds')
        if seconds is not None and not isinstance(seconds, int | float):
            message = f'seconds must be a number, got {seconds!r}'
            raise ValueError(message)
        return cls(
            state=InstanceState(str(document['state'])),
            fingerprint=_optional_str(document=document, key='fingerprint'),
            started_at=_optional_str(document=document, key='started_at'),
            finished_at=_optional_str(document=document, key='finished_at'),
            seconds=None if seconds is None else float(seconds),
            error=_optional_str(document=document, key='error'),
            traceback_file=_optional_str(document=document, key='traceback_file'),
            reason=_optional_str(document=document, key='reason'),
        )


def _optional_str(*, document: Mapping[str, object], key: str) -> str | None:
    value = document.get(key)
    if value is None or isinstance(value, str):
        return value
    message = f'{key} must be a string, got {value!r}'
    raise ValueError(message)


@dataclass(frozen=True, kw_only=True)
class RunStatusSnapshot:
    """An immutable view of ``status.json``.

    Attributes:
        run_state: State of the whole run.
        updated_at: UTC ISO 8601 time of the last write.
        instances: Records keyed by instance id, for example ``"attack[fgsm]"``.
    """

    run_state: RunState
    updated_at: str
    instances: Mapping[str, InstanceRecord] = field(default_factory=dict)

    def with_instance(self, *, instance_id: str, record: InstanceRecord) -> 'RunStatusSnapshot':
        """Return a copy with one instance record added or replaced."""
        instances = dict(self.instances)
        instances[instance_id] = record
        return dataclasses.replace(self, instances=instances, updated_at=utc_now_iso())

    def with_run_state(self, *, run_state: RunState) -> 'RunStatusSnapshot':
        """Return a copy with a new run state."""
        return dataclasses.replace(self, run_state=run_state, updated_at=utc_now_iso())


def utc_now_iso() -> str:
    """Return the current UTC time as ISO 8601 with a ``Z`` suffix, to the second."""
    return datetime.now(tz=UTC).strftime('%Y-%m-%dT%H:%M:%SZ')


def format_instance_id(*, stage_name: str, name_parts: tuple[str, ...]) -> str:
    """Return the display id of an instance, for example ``attack[forecasting/h24/fgsm]``.

    Args:
        stage_name: The stage name.
        name_parts: The validated instance name parts.

    Returns:
        The instance id used in ``status.json``, logs and ``--run.rerun``.
    """
    return f'{stage_name}[{"/".join(name_parts)}]'


def parse_instance_id(*, instance_id: str) -> tuple[str, tuple[str, ...]]:
    """Split an instance id into its stage name and name parts.

    This is the inverse of ``format_instance_id``.

    Args:
        instance_id: For example ``attack[forecasting/h24/fgsm]``.

    Returns:
        ``('attack', ('forecasting', 'h24', 'fgsm'))``.

    Raises:
        ValueError: If ``instance_id`` does not have the form ``stage[name]``.
    """
    stage_name, bracket, rest = instance_id.partition('[')
    if not bracket or not rest.endswith(']') or not stage_name:
        message = f'invalid instance id {instance_id!r}; expected stage[name]'
        raise ValueError(message)
    return stage_name, tuple(rest[:-1].split('/'))


def read_status(*, path: Path) -> RunStatusSnapshot | None:
    """Read ``status.json``.

    Args:
        path: Location of ``status.json``.

    Returns:
        The snapshot, or ``None`` if the file does not exist yet.

    Raises:
        StatusFileError: If the file is not a JSON object, lacks a required key,
            has an unsupported ``format_version`` or holds a malformed record.
            The message names the file.
    """
    require_path(value=path, parameter='path')
    if not path.exists():
        return None
    try:
        return _snapshot_from_document(path=path, document=read_json_object(path=path))
    except StatusFileError:
        raise
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        message = f'{path.as_posix()} is malformed: {error!r}'
        raise StatusFileError(message) from error


def _snapshot_from_document(*, path: Path, document: Mapping[str, object]) -> RunStatusSnapshot:
    if document.get('format_version') != STATUS_FORMAT_VERSION:
        message = (
            f'{path.as_posix()} has format_version {document.get("format_version")!r}; '
            f'this TSLab version reads only {STATUS_FORMAT_VERSION}'
        )
        raise StatusFileError(message)
    missing = [key for key in ('run_state', 'updated_at', 'instances') if key not in document]
    if missing:
        message = f'{path.as_posix()} is missing required keys: {missing}'
        raise StatusFileError(message)
    instances = document['instances']
    if not isinstance(instances, dict) or not isinstance(document['updated_at'], str):
        message = f'{path.as_posix()} has a malformed instances or updated_at entry'
        raise StatusFileError(message)
    return RunStatusSnapshot(
        run_state=RunState(document['run_state']),
        updated_at=document['updated_at'],
        instances={
            instance_id: InstanceRecord.from_json(document=record)
            for instance_id, record in instances.items()
        },
    )


def write_status(*, path: Path, snapshot: RunStatusSnapshot) -> None:
    """Write ``status.json`` atomically.

    Args:
        path: Location of ``status.json``.
        snapshot: The state to write.
    """
    document = {
        'format_version': STATUS_FORMAT_VERSION,
        'run_state': snapshot.run_state.value,
        'updated_at': snapshot.updated_at,
        'instances': {
            instance_id: record.to_json() for instance_id, record in snapshot.instances.items()
        },
    }
    atomic_write_json(path=path, document=document)

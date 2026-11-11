"""Deterministic fingerprints for stage instances and external files.

A fingerprint is a short hash that identifies exactly what produced a result:
the stage, its version, its settings and the fingerprints of its inputs. If any
of these change, the fingerprint changes and TSLab re-runs the stage instead of
reusing a stale result.

The hash is computed over a *canonical* JSON form of the values, so the same
logical value always produces the same bytes (sorted keys, exact floats,
explicit type names for dataclasses and enums).
"""

from collections.abc import Mapping
import dataclasses
from enum import Enum, StrEnum
import hashlib
import json
from pathlib import Path
from typing import (
    Annotated,
    get_args,
    get_origin,
    get_type_hints,
    Protocol,
    runtime_checkable,
    TYPE_CHECKING,
)

from tslab.lab.core.text import qualified_name

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

__all__ = [
    'FINGERPRINT_FORMAT_VERSION',
    'CarriesFingerprint',
    'Fingerprintable',
    'NotFingerprinted',
    'canonicalize',
    'compute_fingerprint',
    'hash_path_contents',
    'is_not_fingerprinted',
]

FINGERPRINT_FORMAT_VERSION = 3
_FINGERPRINT_LENGTH = 16
_CHUNK_SIZE = 1024 * 1024


class NotFingerprinted:
    """Marker for values that must not influence fingerprints.

    Use it inside ``typing.Annotated`` on dataclass fields or stage parameters
    whose value does not change the result, such as the device or the number of
    data-loading workers::

        num_workers: Annotated[int, NotFingerprinted] = 0
    """


@runtime_checkable
class Fingerprintable(Protocol):
    """Objects that describe themselves for fingerprinting.

    Live objects such as models, trainers and data sources cannot be hashed
    directly. They implement this protocol by returning their configuration
    (usually a frozen dataclass), which is then canonicalized.
    """

    def fingerprint_settings(self) -> object:
        """Return a canonicalizable description of this object."""
        ...


@runtime_checkable
class CarriesFingerprint(Protocol):
    """Values that already know their own fingerprint (artifacts)."""

    @property
    def tslab_fingerprint(self) -> str | None:
        """Return the fingerprint attached to this value, if any."""
        ...


def is_not_fingerprinted(*, annotation: object) -> bool:
    """Return True if ``annotation`` is ``Annotated[..., NotFingerprinted]``.

    Args:
        annotation: A type annotation, possibly ``Annotated``.

    Returns:
        Whether the annotation carries the ``NotFingerprinted`` marker.
    """
    if get_origin(annotation) is not Annotated:
        return False
    return any(
        metadata is NotFingerprinted or isinstance(metadata, NotFingerprinted)
        for metadata in get_args(annotation)[1:]
    )


def _canonicalize_dataclass(*, value: 'DataclassInstance') -> dict[str, object]:
    value_type = type(value)
    type_hints = get_type_hints(value_type, include_extras=True)
    canonical: dict[str, object] = {'__type__': qualified_name(cls=value_type)}
    for field in dataclasses.fields(value):
        if is_not_fingerprinted(annotation=type_hints.get(field.name)):
            continue
        canonical[field.name] = canonicalize(value=getattr(value, field.name))
    return canonical


def _canonicalize_artifact(*, value: CarriesFingerprint) -> object:
    fingerprint = value.tslab_fingerprint
    if fingerprint is None:
        message = (
            f'{type(value).__qualname__} was not produced by a stage and has no fingerprint; '
            f'return it from a stage or use {type(value).__qualname__}.external(path=...)'
        )
        raise TypeError(message)
    return {'__fingerprint__': fingerprint}


def _canonicalize_mapping(*, value: Mapping[object, object]) -> dict[str, object]:
    canonical_mapping: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            message = f'cannot fingerprint mapping with non-str key {key!r}'
            raise TypeError(message)
        canonical_mapping[key] = canonicalize(value=item)
    return canonical_mapping


def _enum_value(*, value: Enum) -> str:
    """The value of a ``StrEnum`` member; any other enum cannot be fingerprinted."""
    if isinstance(value, StrEnum):
        return value.value
    message = (
        f'cannot fingerprint {type(value).__qualname__}.{value.name}; '
        f'configs hold choices as StrEnum'
    )
    raise TypeError(message)


# why: one branch per supported type keeps the rules readable
def canonicalize(*, value: object) -> object:  # noqa: PLR0911
    """Convert ``value`` into a JSON-compatible form with a single spelling.

    Rules:
        - ``None``, ``bool``, ``int``, ``str`` are kept as they are.
        - ``float`` becomes its ``repr`` string, which is exact.
        - ``Path`` becomes its POSIX string; the file content is *not* hashed.
        - ``StrEnum`` becomes its value, so a rename of the class or member keeps the
          fingerprint; any other ``Enum`` raises ``TypeError``.
        - Artifacts are replaced by their fingerprint. An artifact without one
          (created by user code, not by a stage) raises ``TypeError``.
        - ``Fingerprintable`` objects are replaced by their settings.
        - Dataclasses become a dict with a ``__type__`` key plus their fields,
          skipping fields annotated with ``NotFingerprinted``.
        - Lists and tuples become lists; mappings with ``str`` keys become
          dicts (key order is handled by sorted JSON serialization).

    Args:
        value: The value to canonicalize.

    Returns:
        A JSON-compatible value.

    Raises:
        TypeError: If ``value`` has an unsupported type. Ignoring it silently
            would let stale results be reused.
    """
    if isinstance(value, Enum):  # before scalars: a StrEnum member is a str
        return _enum_value(value=value)
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        # why: the repr of a NumPy float is not the repr of the same number as a float
        return repr(float(value))
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, CarriesFingerprint):
        return _canonicalize_artifact(value=value)
    # Fingerprintable comes before the dataclass rule on purpose: a dataclass such as
    # ``RunOptions`` can opt out of the fingerprint by returning ``None`` from
    # ``fingerprint_settings``.
    if isinstance(value, Fingerprintable):
        return canonicalize(value=value.fingerprint_settings())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _canonicalize_dataclass(value=value)
    if isinstance(value, list | tuple):
        return [canonicalize(value=item) for item in value]
    if isinstance(value, Mapping):
        return _canonicalize_mapping(value=value)
    message = (
        f'cannot fingerprint value of type {type(value).__qualname__}; make it Fingerprintable '
        f'(add fingerprint_settings()) or pass its config dataclass instead'
    )
    raise TypeError(message)


def compute_fingerprint(*, document: Mapping[str, object]) -> str:
    """Hash an already canonical document into a 16-character hex fingerprint.

    Args:
        document: JSON-compatible mapping, typically built with ``canonicalize``.

    Returns:
        The first 16 hex characters of the SHA-256 digest.
    """
    payload = json.dumps(document, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:_FINGERPRINT_LENGTH]


def _hash_file(*, path: Path) -> str:
    digest = hashlib.sha256()
    with path.open(mode='rb') as file_handle:
        while chunk := file_handle.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


_content_hash_cache: dict[tuple[str, int, int, int], str] = {}


def _cached_file_hash(*, path: Path) -> str:
    stat = path.stat()
    cache_key = (path.resolve().as_posix(), stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if cache_key not in _content_hash_cache:
        _content_hash_cache[cache_key] = _hash_file(path=path)
    return _content_hash_cache[cache_key]


def hash_path_contents(*, path: Path) -> str:
    """Return the SHA-256 hex digest of a file or the contents of a directory tree.

    For a directory, the digest covers the POSIX path of each file relative to the
    directory plus the content hash of that file, in sorted order, each followed by a
    NUL byte. Results are
    memoized per process, keyed by resolved path, inode, size and modification time,
    so hashing a large checkpoint twice in one run costs nothing.

    Args:
        path: Existing file or directory.

    Returns:
        Hex digest of the contents.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.exists():
        message = f'{path.as_posix()} does not exist'
        raise FileNotFoundError(message)
    if path.is_file():
        return _cached_file_hash(path=path)
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob('*') if item.is_file()):
        # NUL separates the fields, so no file name can run into the hash after it.
        digest.update(file_path.relative_to(path).as_posix().encode('utf-8') + b'\0')
        digest.update(hash_path_contents(path=file_path).encode('ascii') + b'\0')
    return digest.hexdigest()

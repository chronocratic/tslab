"""Artifacts: the saved result of one stage instance.

An artifact knows how to write itself into a folder (``save``) and how to read
itself back (``load``). TSLab attaches an identity to every artifact a stage
returns or loads (which instance produced it and with which fingerprint), so
that downstream stages can include it in their own fingerprint.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
import json
import logging
import math
from pathlib import Path
from typing import cast, ClassVar, NoReturn, override, Self

from tslab.lab.core.files import atomic_write_text, read_json_object
from tslab.lab.core.fingerprint import compute_fingerprint, hash_path_contents
from tslab.lab.core.paths import require_path
from tslab.lab.core.text import qualified_name

__all__ = [
    'Artifact',
    'ArtifactIdentity',
    'Json',
    'Metrics',
    'UnavailableArtifact',
    'UnavailableArtifactError',
    'UnavailableReason',
    'bind_identity',
    'is_available',
]

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class ArtifactIdentity:
    """Where an artifact came from.

    Attributes:
        fingerprint: Fingerprint of the producing instance, or the content
            fingerprint for external artifacts.
        instance_id: Id of the producing instance (``"encode[default]"``), or
            ``None`` for external artifacts.
        execution_id: Random id of the execution that wrote the artifact. It
            changes every time the instance actually runs, so downstream
            instances notice a re-execution even when the fingerprint is equal.
            ``None`` for external artifacts.
        external_path: The path given to ``external``, or ``None``.
    """

    fingerprint: str
    instance_id: str | None = None
    execution_id: str | None = None
    external_path: Path | None = None


class Artifact(ABC):
    """The saved result of one stage instance.

    Subclasses implement ``save`` and ``load``. Everything else (identity,
    external files) is provided here.
    """

    _tslab_identity: ArtifactIdentity | None = None

    @abstractmethod
    def save(self, *, directory: Path) -> None:
        """Write the files of the artifact into ``directory`` (which already exists).

        Args:
            directory: An empty folder owned by this artifact.
        """

    @classmethod
    @abstractmethod
    def load(cls, *, directory: Path) -> Self:
        """Read an artifact previously written by ``save``.

        Args:
            directory: The folder ``save`` wrote into.

        Returns:
            The loaded artifact.
        """

    @property
    def tslab_identity(self) -> ArtifactIdentity | None:
        """Return the identity TSLab attached, or ``None`` for a fresh object."""
        return self._tslab_identity

    @property
    def tslab_fingerprint(self) -> str | None:
        """Return the fingerprint used when this artifact is a stage input."""
        return None if self._tslab_identity is None else self._tslab_identity.fingerprint

    @classmethod
    def external(cls, *, path: Path) -> Self:
        """Bring a file or folder that no stage produced into the cache system.

        The fingerprint is the SHA-256 of the content, so replacing the file at
        the same path makes every downstream instance run again. The file is not
        copied into the run folder.

        Args:
            path: An existing file or folder.

        Returns:
            The loaded artifact, carrying a content fingerprint.

        Raises:
            TypeError: If ``path`` is not a ``Path``.
            FileNotFoundError: If ``path`` does not exist.
        """
        require_path(value=path, parameter='path')
        if not path.exists():
            message = f'{cls.__name__}.external: {path.as_posix()} does not exist'
            raise FileNotFoundError(message)
        artifact = cls._load_external(path=path)
        fingerprint = compute_fingerprint(
            document={
                'external': True,
                'artifact_type': qualified_name(cls=cls),
                'content_sha256': hash_path_contents(path=path),
            }
        )
        bind_identity(
            artifact=artifact,
            identity=ArtifactIdentity(fingerprint=fingerprint, external_path=path),
        )
        return artifact

    @classmethod
    def _load_external(cls, *, path: Path) -> Self:
        """Load an external artifact. Folders use ``load``; subclasses add files.

        Args:
            path: Existing file or folder given to ``external``.

        Returns:
            The loaded artifact.

        Raises:
            ValueError: If ``path`` is a file and the subclass does not support files.
        """
        if path.is_dir():
            return cls.load(directory=path)
        message = f'{cls.__name__}.external expects a folder written by save(), got a file'
        raise ValueError(message)


def bind_identity(*, artifact: Artifact, identity: ArtifactIdentity) -> None:
    """Attach ``identity`` to ``artifact``. Called by TSLab, not by users.

    Args:
        artifact: The artifact returned or loaded by a stage.
        identity: Where it came from.
    """
    artifact._tslab_identity = identity  # noqa: SLF001 # why: TSLab owns this private slot


class Json(Artifact):
    """A JSON-serializable mapping saved as ``data.json``.

    Non-finite floats (NaN, infinity) are not valid JSON. They are written as
    ``null`` with a warning instead of producing a file other tools reject.
    """

    FILE_NAME: ClassVar[str] = 'data.json'

    def __init__(self, *, data: Mapping[str, object]) -> None:
        """Create the artifact.

        Args:
            data: JSON-serializable mapping with ``str`` keys.
        """
        self.data: dict[str, object] = dict(data)

    @override
    def save(self, *, directory: Path) -> None:
        """Write ``data.json`` (sorted keys, indented) atomically."""
        text = json.dumps(
            _replace_non_finite(value=self.data, location='data'),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        atomic_write_text(path=directory / self.FILE_NAME, text=text + '\n')

    @classmethod
    @override
    def load(cls, *, directory: Path) -> Self:
        """Read ``data.json`` from ``directory``."""
        require_path(value=directory, parameter='directory')
        return cls._from_file(path=directory / cls.FILE_NAME)

    @classmethod
    @override
    def _load_external(cls, *, path: Path) -> Self:
        return cls._from_file(path=path) if path.is_file() else cls.load(directory=path)

    @classmethod
    def _from_file(cls, *, path: Path) -> Self:
        return cls(data=read_json_object(path=path))


def _replace_non_finite(*, value: object, location: str) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        _LOGGER.warning('non-finite value %r at %s written as null', value, location)
        return None
    if isinstance(value, Mapping):
        return {
            key: _replace_non_finite(value=item, location=f'{location}.{key}')
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [
            _replace_non_finite(value=item, location=f'{location}[{index}]')
            for index, item in enumerate(value)
        ]
    return value


def _metric_value(*, key: str, value: object) -> float:
    """Return ``value`` as a float, or raise ``TypeError`` if it is not a number."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        message = f'metric {key!r} must be int or float, got {type(value).__name__}'
        raise TypeError(message)
    return float(value)


class Metrics(Json):
    """Named scalar results, saved like ``Json``.

    ``null`` values in the file (non-finite floats at save time) load back as NaN.
    """

    def __init__(self, *, data: Mapping[str, float]) -> None:
        """Create the artifact.

        Args:
            data: Metric name to value.

        Raises:
            TypeError: If a value is not an ``int`` or ``float``.
        """
        for key, value in data.items():
            _metric_value(key=key, value=value)
        super().__init__(data=data)

    @property
    def values(self) -> dict[str, float]:
        """Return the metrics as ``float`` values."""
        return {key: _metric_value(key=key, value=value) for key, value in self.data.items()}

    @classmethod
    @override
    def _from_file(cls, *, path: Path) -> Self:
        raw = read_json_object(path=path)
        # ``__init__`` checks that every value is a number.
        data = {key: math.nan if value is None else value for key, value in raw.items()}
        return cls(data=cast('Mapping[str, float]', data))


class UnavailableReason(StrEnum):
    """Why a stage returned an ``UnavailableArtifact``."""

    FAILED = 'failed'
    SKIPPED = 'skipped'
    DRY_RUN = 'dry run'


class UnavailableArtifactError(RuntimeError):
    """Raised when code reads data from an ``UnavailableArtifact``."""


class UnavailableArtifact(Artifact):
    """Placeholder returned instead of a real artifact.

    Returned when the instance failed, was skipped because an input was
    unavailable, or during a dry run. Passing it to another stage makes that
    stage skip itself. Reading any attribute raises ``UnavailableArtifactError``,
    so user code cannot silently continue with missing data.
    """

    def __init__(self, *, instance_id: str, reason: UnavailableReason, detail: str) -> None:
        """Create the placeholder.

        Args:
            instance_id: Id of the instance that did not produce a result.
            reason: Why the result is unavailable.
            detail: Human-readable explanation, for example the error message.
        """
        self._instance_id = instance_id
        self._reason = reason
        self._detail = detail

    @property
    def unavailable_instance_id(self) -> str:
        """Return the id of the instance that did not produce a result."""
        return self._instance_id

    @property
    def unavailable_reason(self) -> UnavailableReason:
        """Return why the result is unavailable."""
        return self._reason

    def __getattr__(self, attribute: str) -> NoReturn:
        """Raise for every attribute that the placeholder does not define."""
        if attribute.startswith('__'):
            raise AttributeError(attribute)
        message = (
            f'{self.unavailable_instance_id} is unavailable ({self.unavailable_reason}: '
            f'{self._detail}); cannot read {attribute!r}'
        )
        raise UnavailableArtifactError(message)

    @override
    def save(self, *, directory: Path) -> NoReturn:
        """Never called: unavailable results are not saved."""
        message = f'cannot save unavailable result of {self.unavailable_instance_id} to {directory}'
        raise UnavailableArtifactError(message)

    @classmethod
    @override
    def load(cls, *, directory: Path) -> NoReturn:
        """Never called: unavailable results are not saved."""
        message = f'cannot load an unavailable result from {directory}'
        raise UnavailableArtifactError(message)


def is_available(*, artifact: Artifact) -> bool:
    """Return ``False`` if ``artifact`` is a placeholder for a missing result.

    Use it in user code between stages, for example to skip plotting the result
    of a failed transform.

    Args:
        artifact: Any value returned by a stage.

    Returns:
        Whether the artifact holds real data.
    """
    return not isinstance(artifact, UnavailableArtifact)

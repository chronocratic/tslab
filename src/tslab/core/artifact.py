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
import shutil
from typing import cast, ClassVar, NoReturn, override, Self

from safetensors import safe_open
from safetensors.torch import save_file
import torch

from tslab.core.fingerprint import compute_fingerprint, hash_path_contents
from tslab.core.paths import atomic_write_json, atomic_write_text, read_json_object, require_path
from tslab.core.text import qualified_name

__all__ = [
    'Artifact',
    'ArtifactIdentity',
    'Checkpoint',
    'Json',
    'Metrics',
    'Tensors',
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


class Tensors(Artifact):
    """Named tensors saved with safetensors as ``tensors.safetensors``.

    safetensors is used instead of ``torch.save`` because loading a pickle can
    execute arbitrary code, while safetensors stores only raw tensor data.
    """

    FILE_NAME: ClassVar[str] = 'tensors.safetensors'

    def __init__(
        self, *, tensors: Mapping[str, torch.Tensor], metadata: Mapping[str, str] | None = None
    ) -> None:
        """Create the artifact.

        Args:
            tensors: Name to tensor.
            metadata: Optional string metadata stored in the file header.
        """
        self.tensors: dict[str, torch.Tensor] = dict(tensors)
        self.metadata: dict[str, str] = dict(metadata or {})

    @override
    def save(self, *, directory: Path) -> None:
        """Write ``tensors.safetensors``.

        Tensors are detached, moved to CPU, made contiguous and cloned, because
        safetensors refuses non-contiguous tensors and tensors sharing memory.
        """
        require_path(value=directory, parameter='directory')
        prepared = {
            key: tensor.detach().to(device='cpu').contiguous().clone()
            for key, tensor in self.tensors.items()
        }
        save_file(
            tensors=prepared, filename=directory / self.FILE_NAME, metadata=self.metadata or None
        )

    @classmethod
    @override
    def load(cls, *, directory: Path) -> Self:
        """Read ``tensors.safetensors`` from ``directory``."""
        require_path(value=directory, parameter='directory')
        return cls._from_file(path=directory / cls.FILE_NAME)

    @classmethod
    @override
    def _load_external(cls, *, path: Path) -> Self:
        return cls._from_file(path=path) if path.is_file() else cls.load(directory=path)

    @classmethod
    def _from_file(cls, *, path: Path) -> Self:
        with safe_open(path, framework='pt') as handle:
            metadata = handle.metadata() or {}
            tensors = {key: handle.get_tensor(key) for key in handle.keys()}  # noqa: SIM118
        return cls(tensors=tensors, metadata=metadata)


class Checkpoint(Artifact):
    """Names the best and the last training checkpoint inside a folder.

    The trainer writes the checkpoint files into the folder while it trains;
    ``save`` only writes ``checkpoint.json`` with their names, so large files
    are never copied. A training checkpoint is a pickle, and loading a pickle
    can execute code, so ``trusted`` says whether full unpickling is allowed:
    checkpoints written by a stage in this run folder are trusted, files given
    to ``external`` are not unless the caller says so. The flag is saved in
    ``checkpoint.json``, so an untrusted file stays untrusted when a later run
    loads the cached result.
    """

    FILE_NAME: ClassVar[str] = 'checkpoint.json'
    FORMAT_VERSION: ClassVar[int] = 2

    def __init__(self, *, directory: Path, best: str, last: str, trusted: bool = True) -> None:
        """Create the artifact.

        Args:
            directory: Folder holding the checkpoint files.
            best: File name of the best checkpoint.
            last: File name of the last checkpoint.
            trusted: Whether the files may be fully unpickled when loaded.

        Raises:
            TypeError: If ``directory`` is not a ``Path``.
            ValueError: If a name is not a plain file name.
        """
        require_path(value=directory, parameter='directory')
        for name in (best, last):
            _require_file_name(name=name)
        self.directory = directory
        self.best = best
        self.last = last
        self.trusted = trusted

    @property
    def best_path(self) -> Path:
        """Return the path of the best checkpoint file."""
        return self.directory / self.best

    @property
    def last_path(self) -> Path:
        """Return the path of the last checkpoint file."""
        return self.directory / self.last

    @override
    def save(self, *, directory: Path) -> None:
        """Write ``checkpoint.json``; copy the files first if ``directory`` is another folder.

        After a copy this checkpoint points at ``directory``, where the files now live.
        """
        require_path(value=directory, parameter='directory')
        if directory.resolve() != self.directory.resolve():
            for name in sorted({self.best, self.last}):
                shutil.copy2(self.directory / name, directory / name)
            self.directory = directory
        document = {
            'best': self.best,
            'format_version': self.FORMAT_VERSION,
            'last': self.last,
            'trusted': self.trusted,
        }
        atomic_write_json(path=directory / self.FILE_NAME, document=document)

    @classmethod
    @override
    def load(cls, *, directory: Path) -> Self:
        """Read ``checkpoint.json`` and check that both named files exist.

        Raises:
            ValueError: If ``checkpoint.json`` has another format version or bad names.
            FileNotFoundError: If a named checkpoint file is missing.
        """
        require_path(value=directory, parameter='directory')
        document = read_json_object(path=directory / cls.FILE_NAME)
        if document.get('format_version') != cls.FORMAT_VERSION:
            message = (
                f'{(directory / cls.FILE_NAME).as_posix()}: expected format_version '
                f'{cls.FORMAT_VERSION}'
            )
            raise ValueError(message)
        best = document.get('best')
        last = document.get('last')
        trusted = document.get('trusted')
        if not isinstance(best, str) or not isinstance(last, str) or not isinstance(trusted, bool):
            message = (
                f'{(directory / cls.FILE_NAME).as_posix()}: best and last must be strings '
                f'and trusted a boolean'
            )
            raise ValueError(message)  # noqa: TRY004 # why: bad file content, not arguments
        checkpoint = cls(directory=directory, best=best, last=last, trusted=trusted)
        missing = [name for name in (best, last) if not (directory / name).is_file()]
        if missing:
            message = f'checkpoint files missing in {directory.as_posix()}: {missing}'
            raise FileNotFoundError(message)
        return checkpoint

    @classmethod
    @override
    def external(cls, *, path: Path, trusted: bool = False) -> Self:
        """Use a checkpoint file (or a folder with ``checkpoint.json``) from outside the run.

        Args:
            path: A ``.ckpt`` file, used as both best and last, or a folder
                written by ``save``.
            trusted: Allow full unpickling when the weights are loaded. Only
                set it for files you trust: unpickling can execute code.

        Returns:
            The checkpoint, fingerprinted by the content of ``path``.
        """
        checkpoint = super().external(path=path)
        checkpoint.trusted = trusted
        return checkpoint

    @classmethod
    @override
    def _load_external(cls, *, path: Path) -> Self:
        if path.is_dir():
            return cls.load(directory=path)
        return cls(directory=path.parent, best=path.name, last=path.name)


def _require_file_name(*, name: str) -> None:
    """Raise ``ValueError`` unless ``name`` is a plain file name inside its folder."""
    if name in {'', '.', '..'} or Path(name).name != name:
        message = f'checkpoint file name must be a plain file name, got {name!r}'
        raise ValueError(message)


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

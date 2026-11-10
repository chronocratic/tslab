"""Artifacts that hold tensors and training checkpoints."""

from collections.abc import Mapping
from pathlib import Path
import shutil
from typing import ClassVar, override, Self

from safetensors import safe_open
from safetensors.torch import save_file
import torch

from tslab.lab.core.artifact import Artifact
from tslab.lab.core.files import atomic_write_json, read_json_object
from tslab.lab.core.paths import require_path

__all__ = ['Checkpoint', 'Tensors']


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

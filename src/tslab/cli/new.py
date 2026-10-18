"""``tslab new``: start a research project from the TSLab template.

The template is a set of plain files shipped inside the package
(``tslab/templates/project``). They are copied with the standard library, and
two placeholders are replaced in file names and contents: ``{{project_name}}``
(the distribution name, for example ``my-paper``) and ``{{package_name}}``
(the import name, ``my_paper``). The copy belongs to the user from then on.
"""

from dataclasses import dataclass
from importlib import resources
import keyword
import logging
from pathlib import Path
import re
from typing import Annotated

import tyro

from tslab.core.paths import require_path

__all__ = ['New', 'create_project']

_LOGGER = logging.getLogger(__name__)
_NAME_PATTERN = re.compile(r'[A-Za-z][A-Za-z0-9-]*')
_SKIPPED_PARTS = frozenset({'__pycache__'})


@dataclass(frozen=True, kw_only=True)
class New:
    """Create a project from the TSLab template.

    Attributes:
        path: Folder to create; it must not exist or must be empty.
        name: Project name (letters, digits and dashes); defaults to the folder name.
    """

    path: Annotated[Path, tyro.conf.Positional]
    name: str | None = None

    def run(self) -> int:
        """Create the project and print the next steps.

        Returns:
            ``0`` on success, ``1`` if the folder or the name is not usable.
        """
        try:
            package_name = create_project(path=self.path, name=self.name)
        except ValueError as error:
            failure = error  # logged below, without a traceback: it is a usage error
        else:
            _LOGGER.info('created %s', self.path.as_posix())
            _LOGGER.info('next steps:')
            _LOGGER.info('  cd %s', self.path.as_posix())
            _LOGGER.info('  uv sync')
            _LOGGER.info('  uv run python -m %s.toy_data data', package_name)
            _LOGGER.info('  then follow README.md')
            return 0
        _LOGGER.error('%s', failure)
        return 1


def create_project(*, path: Path, name: str | None = None) -> str:
    """Copy the project template to ``path``.

    Args:
        path: Folder to create; it must not exist or must be empty.
        name: Project name; defaults to ``path.name``.

    Returns:
        The package (import) name, ``name`` with dashes turned into underscores.

    Raises:
        TypeError: If ``path`` is not a ``pathlib.Path``.
        ValueError: If ``path`` is a non-empty folder or a file, or the name
            is not letters, digits and dashes starting with a letter, or the
            package name is a Python keyword.
    """
    require_path(value=path, parameter='path')
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        message = f'{path.as_posix()} already exists and is not an empty folder'
        raise ValueError(message)
    project_name = path.name if name is None else name
    if not _NAME_PATTERN.fullmatch(project_name):
        message = (
            f'project name {project_name!r} must start with a letter and contain only '
            'letters, digits and dashes; pass --name'
        )
        raise ValueError(message)
    package_name = project_name.replace('-', '_').lower()
    if keyword.iskeyword(package_name):
        message = (
            f'project name {project_name!r} would make the package {package_name!r}, '
            'a Python keyword that cannot be imported; pass another --name'
        )
        raise ValueError(message)
    replacements = {'{{project_name}}': project_name, '{{package_name}}': package_name}
    with resources.as_file(resources.files('tslab') / 'templates' / 'project') as template:
        for source in sorted(template.rglob('*')):
            relative = source.relative_to(template)
            if not source.is_file() or _SKIPPED_PARTS.intersection(relative.parts):
                continue
            target = path / _replace(text=relative.as_posix(), replacements=replacements)
            target.parent.mkdir(parents=True, exist_ok=True)
            content = source.read_text(encoding='utf-8')
            target.write_text(_replace(text=content, replacements=replacements), encoding='utf-8')
    return package_name


def _replace(*, text: str, replacements: dict[str, str]) -> str:
    for placeholder, value in replacements.items():
        text = text.replace(placeholder, value)
    return text

"""``tslab new``: start a research project from a TSLab template.

A template is a set of plain files shipped inside the package. They are copied
with the standard library, and two placeholders are replaced in file names and
contents: ``{{project_name}}`` (the distribution name, for example ``my-paper``)
and ``{{package_name}}`` (the import name, ``my_paper``). The copy belongs to
the user from then on. The lab ships the ``lab`` template; layers above it add
theirs with ``register_template``, inside ``template_scope``.
"""

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import resources
from importlib.resources.abc import Traversable
import keyword
import logging
from pathlib import Path
import re
from typing import Annotated

import tyro

from tslab.lab.core.paths import require_path

__all__ = [
    'LAB_TEMPLATE',
    'New',
    'ProjectTemplate',
    'create_project',
    'register_template',
    'template_scope',
]

_LOGGER = logging.getLogger(__name__)
_NAME_PATTERN = re.compile(r'[A-Za-z][A-Za-z0-9-]*')
_SKIPPED_PARTS = frozenset({'__pycache__'})


@dataclass(frozen=True, kw_only=True)
class ProjectTemplate:
    """A project template: where its files are and what to do after copying them.

    Attributes:
        name: Value of ``--template`` that picks it.
        files: Folder with the template files, placeholders included.
        next_steps: Commands printed after ``cd <path>``; ``{{package_name}}`` is filled in.
    """

    name: str
    files: Traversable
    next_steps: tuple[str, ...]


LAB_TEMPLATE = ProjectTemplate(
    name='lab',
    files=resources.files('tslab.lab') / 'templates' / 'project',
    next_steps=('uv sync', 'uv run pytest'),
)
_TEMPLATES: dict[str, ProjectTemplate] = {LAB_TEMPLATE.name: LAB_TEMPLATE}
_DEFAULT_TEMPLATE = LAB_TEMPLATE.name


@contextmanager
def template_scope() -> Generator[None]:
    """Keep the templates registered inside the block from outliving it."""
    global _DEFAULT_TEMPLATE  # noqa: PLW0603 # why: restored together with the registry
    saved, saved_default = dict(_TEMPLATES), _DEFAULT_TEMPLATE
    try:
        yield
    finally:
        _TEMPLATES.clear()
        _TEMPLATES.update(saved)
        _DEFAULT_TEMPLATE = saved_default


def register_template(*, template: ProjectTemplate, default: bool = False) -> None:
    """Offer a template to ``tslab new --template``.

    Registering the same template again does nothing.

    Args:
        template: The template to offer.
        default: Make it the template ``tslab new`` uses without ``--template``.

    Raises:
        ValueError: If the name is taken by a different template.
    """
    if _TEMPLATES.get(template.name, template) != template:
        message = f'template {template.name!r} is already registered'
        raise ValueError(message)
    _TEMPLATES[template.name] = template
    if default:
        global _DEFAULT_TEMPLATE  # noqa: PLW0603 # why: the one default template of the process
        _DEFAULT_TEMPLATE = template.name


@dataclass(frozen=True, kw_only=True)
class New:
    """Create a project from a TSLab template.

    For example:
        tslab new my-paper
        tslab new my-paper --template lab

    Attributes:
        path: Folder to create; it must not exist or must be empty.
        template: Template name; the default template when not given.
        name: Project name (letters, digits and dashes); defaults to the folder name.
    """

    path: Annotated[Path, tyro.conf.arg(metavar='FOLDER'), tyro.conf.Positional]
    name: Annotated[str | None, tyro.conf.arg(metavar='NAME', help_behavior_hint='')] = None
    template: Annotated[str | None, tyro.conf.arg(metavar='TEMPLATE', help_behavior_hint='')] = None

    def run(self) -> int:
        """Create the project and print the next steps.

        Returns:
            ``0`` on success, ``1`` if the folder, the name or the template is not usable.
        """
        chosen = self.template if self.template is not None else _DEFAULT_TEMPLATE
        if chosen not in _TEMPLATES:
            _LOGGER.error('unknown template %s; choose from %s', chosen, ', '.join(_TEMPLATES))
            return 1
        template = _TEMPLATES[chosen]
        try:
            package_name = create_project(path=self.path, template=template, name=self.name)
        except ValueError as error:
            failure = error  # logged below, without a traceback: it is a usage error
        else:
            _LOGGER.info('created %s', self.path.as_posix())
            _LOGGER.info('next steps:')
            _LOGGER.info('  cd %s', self.path.as_posix())
            for step in template.next_steps:
                _LOGGER.info('  %s', step.replace('{{package_name}}', package_name))
            return 0
        _LOGGER.error('%s', failure)
        return 1


def create_project(*, path: Path, template: ProjectTemplate, name: str | None = None) -> str:
    """Copy a project template to ``path``.

    Args:
        path: Folder to create; it must not exist or must be empty.
        template: The template to copy.
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
    with resources.as_file(template.files) as root:
        for source in sorted(root.rglob('*')):
            _copy_template_file(source=source, root=root, path=path, replacements=replacements)
    return package_name


def _copy_template_file(
    *, source: Path, root: Path, path: Path, replacements: dict[str, str]
) -> None:
    """Copy one template file with its placeholders replaced; folders and skipped parts are left."""
    relative = source.relative_to(root)
    if not source.is_file() or _SKIPPED_PARTS.intersection(relative.parts):
        return
    target = path / _replace(text=relative.as_posix(), replacements=replacements)
    target.parent.mkdir(parents=True, exist_ok=True)
    content = source.read_text(encoding='utf-8')
    target.write_text(_replace(text=content, replacements=replacements), encoding='utf-8')


def _replace(*, text: str, replacements: dict[str, str]) -> str:
    for placeholder, value in replacements.items():
        text = text.replace(placeholder, value)
    return text

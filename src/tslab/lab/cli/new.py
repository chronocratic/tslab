"""``tslab new``: start a research project from a TSLab template.

A template is a set of plain files shipped inside the package. They are copied
with the standard library, and two placeholders are replaced in file names and
contents: ``{{project_name}}`` (the distribution name, for example ``my-paper``)
and ``{{package_name}}`` (the import name, ``my_paper``). The copy belongs to
the user from then on. The lab ships the ``lab`` template; layers above it add
theirs through the ``templates`` of the command line entry point.

The layout is picked from the surroundings. Inside a Python project (a folder
above holds a ``pyproject.toml``) the new folder is a uv workspace member of
it: still a full project, plus the lines in both ``pyproject.toml`` files that
link the two. Elsewhere it is a standalone project. ``--layout`` overrides the
choice. The parent file is only appended to, with the standard library; a table
the parent already has is never edited, the lines to add by hand are logged.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from importlib import resources
from importlib.resources.abc import Traversable
import keyword
import logging
from pathlib import Path
import re
import tomllib
from typing import Annotated, Any

import tyro

from tslab.lab.core.paths import require_path

__all__ = ['LAB_TEMPLATE', 'Layout', 'New', 'ProjectTemplate', 'create_project']

_LOGGER = logging.getLogger(__name__)
_NAME_PATTERN = re.compile(r'[A-Za-z][A-Za-z0-9-]*')
_SKIPPED_PARTS = frozenset({'__pycache__'})
_DEPENDENCIES = re.compile(r'^dependencies = \[', flags=re.MULTILINE)
# Per table of the parent: its key path, the block to append when it is absent, and the
# instruction to log when it exists. A table that exists is never edited.
type _ParentEdit = tuple[tuple[str, ...], str, str]
_WORKSPACE_EDITS: tuple[_ParentEdit, ...] = (
    (
        ('tool', 'uv', 'workspace'),
        '[tool.uv.workspace]\nmembers = ["{relative}"]',
        'add "{relative}" to members of [tool.uv.workspace]',
    ),
)
_DEPENDENCY_EDITS: tuple[_ParentEdit, ...] = (
    (
        ('tool', 'uv', 'sources'),
        '[tool.uv.sources]\n{member} = {{ workspace = true }}',
        "add '{member} = {{ workspace = true }}' to [tool.uv.sources]",
    ),
    (
        ('dependency-groups',),
        '[dependency-groups]\ndev = ["{member}"]',
        'add "{member}" to the dev group of [dependency-groups]',
    ),
)


class Layout(StrEnum):
    """How a new project relates to the folders around it.

    Attributes:
        STANDALONE: A full project on its own.
        WORKSPACE: A full project that is also a uv workspace member of the enclosing one.
    """

    STANDALONE = 'standalone'
    WORKSPACE = 'workspace'


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


@dataclass(frozen=True, kw_only=True)
class New:
    """Create a project from a TSLab template.

    For example:
        tslab new my-paper
        tslab new my-paper --template lab
        tslab new my-paper --layout standalone

    Attributes:
        path: Folder to create; it must not exist or must be empty.
        template: Template name; the default template when not given.
        name: Project name (letters, digits and dashes); defaults to the folder name.
        layout: 'workspace' links the project into the enclosing one as a uv workspace
            member, 'standalone' does not; picked from the surroundings when not given.
    """

    path: Annotated[Path, tyro.conf.arg(metavar='FOLDER'), tyro.conf.Positional]
    name: Annotated[str | None, tyro.conf.arg(metavar='NAME', help_behavior_hint='')] = None
    template: Annotated[str | None, tyro.conf.arg(metavar='TEMPLATE', help_behavior_hint='')] = None
    layout: Annotated[
        Layout | None,
        tyro.conf.arg(metavar='{standalone,workspace}', help_behavior_hint=''),
        tyro.conf.EnumChoicesFromValues,
    ] = None

    def run(
        self,
        *,
        templates: Sequence[ProjectTemplate] = (LAB_TEMPLATE,),
        default_template: str = LAB_TEMPLATE.name,
    ) -> int:
        """Create the project and print the next steps.

        Args:
            templates: The templates on offer; a layer above the lab adds its own.
            default_template: Name of the template used without ``--template``.

        Returns:
            ``0`` on success, ``1`` if the folder, the name, the template, the layout or the
            enclosing project file is not usable.
        """
        chosen = self.template if self.template is not None else default_template
        offered = {template.name: template for template in templates}
        if chosen not in offered:
            _LOGGER.error('unknown template %s; choose from %s', chosen, ', '.join(offered))
            return 1
        template = offered[chosen]
        try:
            package_name = create_project(
                path=self.path, template=template, name=self.name, layout=self.layout
            )
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


def create_project(
    *, path: Path, template: ProjectTemplate, name: str | None = None, layout: Layout | None = None
) -> str:
    """Copy a project template to ``path``, linking it into the enclosing project if any.

    Args:
        path: Folder to create; it must not exist or must be empty.
        template: The template to copy.
        name: Project name; defaults to ``path.name``.
        layout: ``Layout.WORKSPACE`` or ``Layout.STANDALONE``; ``None`` picks workspace when a
            folder above ``path`` holds a ``pyproject.toml`` and standalone otherwise.

    Returns:
        The package (import) name, ``name`` with dashes turned into underscores.

    Raises:
        TypeError: If ``path`` is not a ``pathlib.Path``.
        ValueError: If ``path`` is a non-empty folder or a file, or the name
            is not letters, digits and dashes starting with a letter, or the
            package name is a Python keyword, or the layout is workspace without an
            enclosing project, or the enclosing ``pyproject.toml`` does not parse, or the
            enclosing project has the same name. Nothing is written in these cases.
    """
    require_path(value=path, parameter='path')
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        message = f'{path.as_posix()} already exists and is not an empty folder'
        raise ValueError(message)
    project_name = path.name if name is None else name
    package_name = _package_name(project_name=project_name)
    parent = _resolve_parent(path=path, layout=layout, project_name=project_name)
    replacements = {'{{project_name}}': project_name, '{{package_name}}': package_name}
    with resources.as_file(template.files) as root:
        for source in sorted(root.rglob('*')):
            _copy_template_file(source=source, root=root, path=path, replacements=replacements)
    if parent is None:
        _LOGGER.info('created %s as a standalone project', project_name)
        return package_name
    _link_workspace_member(path=path, parent=parent, member=project_name)
    return package_name


def _package_name(*, project_name: str) -> str:
    """Return the import name of a project, or raise if the name is not usable."""
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
    return package_name


def _resolve_parent(*, path: Path, layout: Layout | None, project_name: str) -> Path | None:
    """Return the folder to link into, or ``None`` for a standalone project.

    Raises:
        ValueError: If the layout needs a parent that is missing, or the parent file does not
            parse, or the parent has the same name as the new project.
    """
    if layout is Layout.STANDALONE:
        return None
    parent = _find_parent(path=path)
    if parent is None and layout is Layout.WORKSPACE:
        message = f'--layout workspace needs a pyproject.toml in a folder above {path.as_posix()}'
        raise ValueError(message)
    if parent is None:
        return None
    parent_name = _parent_name(parent=parent)
    if parent_name is None or _normalize(parent_name) != _normalize(project_name):
        return parent
    message = (
        f'project name {project_name!r} is the name of the enclosing project in '
        f'{parent.as_posix()}; pass another --name'
    )
    raise ValueError(message)


def _normalize(name: str) -> str:
    return re.sub(r'[-_.]+', '-', name).lower()


def _find_parent(*, path: Path) -> Path | None:
    """Return the nearest folder above ``path`` that holds a ``pyproject.toml``."""
    return next(
        (folder for folder in path.resolve().parents if (folder / 'pyproject.toml').is_file()), None
    )


def _read_parent(*, parent: Path) -> tuple[str, dict[str, Any]]:
    file = parent / 'pyproject.toml'
    text = file.read_text(encoding='utf-8')
    try:
        return text, tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        message = f'cannot read {file.as_posix()}: {error}; fix it or pass --layout standalone'
        raise ValueError(message) from error


def _has_table(*, data: dict[str, Any], keys: Sequence[str]) -> bool:
    """Tell whether the nested key path ``keys`` exists in ``data``."""
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return False
        current = current[key]
    return True


def _parent_name(*, parent: Path) -> str | None:
    project = _read_parent(parent=parent)[1].get('project')
    name = project.get('name') if isinstance(project, dict) else None
    return name if isinstance(name, str) else None


def _parent_edits(
    *, data: dict[str, Any], member: str, relative: str, edit_dependencies: bool
) -> tuple[list[str], list[str]]:
    """Split the wanted parent edits into tables to append and lines to add by hand."""
    specs = _WORKSPACE_EDITS + (_DEPENDENCY_EDITS if edit_dependencies else ())
    appended: list[str] = []
    by_hand: list[str] = []
    for keys, block, hand in specs:
        missing = not _has_table(data=data, keys=keys)
        target = appended if missing else by_hand
        target.append((block if missing else hand).format(member=member, relative=relative))
    return appended, by_hand


def _append_tables(*, file: Path, original: str, blocks: Sequence[str]) -> bool:
    """Append ``blocks`` to ``file``; put ``original`` back and return ``False`` if it breaks."""
    if not blocks:
        return True
    text = original if original.endswith('\n') else original + '\n'
    text += ''.join(f'\n{block}\n' for block in blocks)
    file.write_text(text, encoding='utf-8')
    try:
        tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        file.write_text(original, encoding='utf-8')
        return False
    return True


def _link_workspace_member(*, path: Path, parent: Path, member: str) -> None:
    """Make the new project a uv workspace member of ``parent``."""
    file = parent / 'pyproject.toml'
    original, data = _read_parent(parent=parent)
    parent_name = _parent_name(parent=parent)
    appended, by_hand = _parent_edits(
        data=data,
        member=member,
        relative=path.resolve().relative_to(parent).as_posix(),
        edit_dependencies=parent_name is not None,
    )
    if parent_name is not None:
        _link_to_parent(path=path / 'pyproject.toml', parent_name=parent_name)
    if not _append_tables(file=file, original=original, blocks=appended):
        by_hand = [f'append {block!r}' for block in appended] + by_hand
    _LOGGER.info('created %s as a uv workspace member of %s', member, parent_name or parent.name)
    if by_hand:
        _LOGGER.warning('edit %s by hand: %s', file.as_posix(), '; '.join(by_hand))


def _link_to_parent(*, path: Path, parent_name: str) -> None:
    """Make the member project at ``path`` depend on the parent as a workspace source."""
    text = path.read_text(encoding='utf-8')
    text = _DEPENDENCIES.sub(f'dependencies = ["{parent_name}", ', text, count=1)
    text += f'\n[tool.uv.sources]\n{parent_name} = {{ workspace = true }}\n'
    path.write_text(text, encoding='utf-8')


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

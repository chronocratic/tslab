"""Static dependency rules for the movable jobs package."""

import ast
from pathlib import Path
import sys

_ROOT = Path(__file__).parents[2]
_SOURCE = _ROOT / 'src'
_TESTS = _ROOT / 'tests'
_JOBS_SOURCE = _SOURCE / 'tslab' / 'jobs'
_JOBS_TESTS = _TESTS / 'unit' / 'jobs'


def _python_files(*, root: Path) -> list[Path]:
    return sorted(root.rglob('*.py')) if root.is_dir() else []


def _module_name(*, path: Path, root: Path, package: str) -> str:
    parts = list(path.relative_to(root).with_suffix('').parts)
    if parts[-1] == '__init__':
        parts.pop()
    return '.'.join((package, *parts)) if parts else package


def _imports(*, path: Path, root: Path, package: str) -> list[tuple[int, str]]:
    module_name = _module_name(path=path, root=root, package=package)
    current_package = module_name if path.name == '__init__.py' else module_name.rpartition('.')[0]
    targets: list[tuple[int, str]] = []
    source = path.read_text().replace('{{package_name}}', 'template_package')
    source = source.replace('{{project_name}}', 'template_project')
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            target = node.module or ''
            if node.level:
                parts = current_package.split('.')[
                    : len(current_package.split('.')) - node.level + 1
                ]
                target = '.'.join((*parts, target) if target else parts)
            targets.append((node.lineno, target))
    return targets


def _is_within(*, path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def test_jobs_package_imports_only_stdlib_and_itself() -> None:
    violations = [
        f'{path.relative_to(_ROOT)}:{line}:{target}'
        for path in _python_files(root=_JOBS_SOURCE)
        for line, target in _imports(path=path, root=_JOBS_SOURCE, package='tslab.jobs')
        if target.split('.', 1)[0] not in sys.stdlib_module_names
        and target != '__future__'
        and not (target == 'tslab.jobs' or target.startswith('tslab.jobs.'))
    ]
    assert not violations, '\n'.join(violations)


def test_jobs_package_never_says_tslab() -> None:
    violations = []
    for path in _python_files(root=_JOBS_SOURCE):
        text = path.read_text().replace('tslab.jobs', '')
        violations.extend(
            f'{path.relative_to(_ROOT)}:{line}:tslab'
            for line, content in enumerate(text.splitlines(), start=1)
            if 'tslab' in content.casefold()
        )
    assert not violations, '\n'.join(violations)


def test_code_outside_imports_only_the_jobs_package_root() -> None:
    violations = [
        f'{path.relative_to(_ROOT)}:{line}:{target}'
        for search_root in (_SOURCE, _TESTS)
        for path in _python_files(root=search_root)
        if not _is_within(path=path, root=_JOBS_SOURCE)
        and not _is_within(path=path, root=_JOBS_TESTS)
        for line, target in _imports(
            path=path,
            root=_SOURCE if _is_within(path=path, root=_SOURCE) else _TESTS,
            package='tslab' if _is_within(path=path, root=_SOURCE) else 'tests',
        )
        if target.startswith('tslab.jobs.')
    ]
    assert not violations, '\n'.join(violations)


def test_jobs_tests_import_only_the_jobs_package() -> None:
    allowed_roots = sys.stdlib_module_names | {'pytest'}
    violations = [
        f'{path.relative_to(_ROOT)}:{line}:{target}'
        for path in _python_files(root=_JOBS_TESTS)
        for line, target in _imports(path=path, root=_JOBS_TESTS, package='tests.unit.jobs')
        if target.split('.', 1)[0] not in allowed_roots
        and not (target == 'tslab.jobs' or target.startswith('tslab.jobs.'))
        and not (target == 'tests.unit.jobs' or target.startswith('tests.unit.jobs.'))
    ]
    assert not violations, '\n'.join(violations)

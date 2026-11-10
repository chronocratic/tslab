"""No re-exports outside package doors (CODING_RULES Part III, rules 48 and 49).

A module that is not an ``__init__.py`` lists in ``__all__`` only names it
defines, and an import names the module that defines the thing or a package
door (an ``__init__.py``).
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'src' / 'tslab'
TEMPLATE = SOURCE / 'ts' / 'templates'
SEARCH_ROOTS = (ROOT / 'src', ROOT / 'tests', ROOT / 'examples')


def _in_template(*, path: Path) -> bool:
    # why: the template files hold {{placeholders}} that do not parse
    return path.is_relative_to(TEMPLATE)


def _parse(*, path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding='utf-8'), filename=str(path))


def _module_statements(*, tree: ast.Module) -> list[ast.stmt]:
    """Module-level statements, including the branches of module-level if and try."""
    statements: list[ast.stmt] = list(tree.body)
    for statement in tree.body:
        if isinstance(statement, (ast.If, ast.Try)):
            statements.extend(statement.body)
            statements.extend(statement.orelse)
            if isinstance(statement, ast.Try):
                for handler in statement.handlers:
                    statements.extend(handler.body)
    return statements


def _module_names(*, tree: ast.Module) -> frozenset[str]:
    """The names a module defines: functions, classes, assignments and type aliases."""
    names: set[str] = set()
    for statement in _module_statements(tree=tree):
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(statement.name)
        elif isinstance(statement, ast.TypeAlias):
            names.add(statement.name.id)
        elif isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            names.add(statement.target.id)
    return frozenset(names)


def _all_entries(*, tree: ast.Module) -> list[str]:
    """The string entries of the module's ``__all__``."""
    for statement in tree.body:
        if not isinstance(statement, ast.Assign):
            continue
        targets = [target for target in statement.targets if isinstance(target, ast.Name)]
        if not any(target.id == '__all__' for target in targets):
            continue
        if not isinstance(statement.value, (ast.List, ast.Tuple)):
            return []
        return [
            entry.value
            for entry in statement.value.elts
            if isinstance(entry, ast.Constant) and isinstance(entry.value, str)
        ]
    return []


def _defined_names(*, module_path: Path) -> frozenset[str] | None:
    """The names of ``module_path.py``, or ``None`` when the module is missing."""
    module_file = module_path.with_suffix('.py')
    if not module_file.is_file():
        return None
    return _module_names(tree=_parse(path=module_file))


def test_modules_export_only_their_own_names() -> None:
    offenders: list[str] = []
    for path in sorted(SOURCE.rglob('*.py')):
        if path.name == '__init__.py' or _in_template(path=path):
            continue
        tree = _parse(path=path)
        own = _module_names(tree=tree)
        offenders.extend(
            f'{path.relative_to(ROOT)}: {entry}'
            for entry in _all_entries(tree=tree)
            if entry not in own
        )
    message = 'a module exports a name it does not define: ' + ', '.join(offenders)
    assert not offenders, message


def _is_subpart(*, module_path: Path, name: str) -> bool:
    """Whether ``name`` is a submodule or subpackage of ``module_path``."""
    return (module_path / f'{name}.py').is_file() or (module_path / name / '__init__.py').is_file()


def _import_detail(*, module: str, alias: ast.alias) -> str | None:
    """The violation text for one imported name, or ``None`` when the import is fine."""
    if module != 'tslab' and not module.startswith('tslab.'):
        return None  # only absolute imports of TSLab are checked
    parts = module.split('.')
    module_path = SOURCE if parts == ['tslab'] else SOURCE.joinpath(*parts[1:])
    if (module_path / '__init__.py').is_file():
        return None  # a package door may re-export
    if alias.name == '*':
        return f'* from {module}'
    names = _defined_names(module_path=module_path)
    if names is None:
        return f'{alias.name} from {module} (module not found)'
    if alias.name in names or _is_subpart(module_path=module_path, name=alias.name):
        return None
    return f'{alias.name} from {module}'


def test_imports_name_the_defining_module() -> None:
    offenders: list[str] = []
    for root in SEARCH_ROOTS:
        for path in sorted(root.rglob('*.py')):
            if _in_template(path=path):
                continue
            tree = _parse(path=path)
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or node.level != 0:
                    continue
                for alias in node.names:
                    detail = _import_detail(module=node.module or '', alias=alias)
                    if detail is not None:
                        offenders.append(f'{path.relative_to(ROOT)}:{node.lineno}: {detail}')
    message = 'an import names a module that does not define the thing: ' + ', '.join(offenders)
    assert not offenders, message

"""``tslab.core`` is the bottom layer: it imports neither Lightning nor higher layers."""

import ast
from pathlib import Path

import pytest

import tslab.core

CORE = Path(tslab.core.__file__).parent
FORBIDDEN = ('lightning', 'tslab.interfaces', 'tslab.tracking', 'tslab.stages', 'tslab.lightning')


def imported_modules(*, path: Path) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


@pytest.mark.parametrize('path', sorted(CORE.glob('*.py')), ids=lambda path: path.name)
def test_core_module_imports_only_lower_layers(path: Path) -> None:
    offending = {
        module
        for module in imported_modules(path=path)
        if any(module == name or module.startswith(f'{name}.') for name in FORBIDDEN)
    }
    assert not offending

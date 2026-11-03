"""SLURM and local never share a file, and the code never names a backend.

The backend is picked once, through the ``BACKENDS`` table of
``study/launchers/__init__.py`` and the ``PROFILE_BACKENDS`` table of
``study/machine/load.py``. After that, the code calls the methods of the
backend it was given; it never compares a backend name or
``.launcher``, and one module never imports both a SLURM module and a
local one. Each rule pins its current counts per file in
``backend_baseline.json``: a count above the baseline fails, and a count
below it fails until the baseline is lowered in the same commit, so the
baseline only ever goes down (the plan's shrinking-baseline guard).
"""

import ast
from collections.abc import Mapping, Sequence
from importlib.util import resolve_name
import itertools
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'src' / 'tslab'
TEMPLATE = SOURCE / 'templates'
UNIT = ROOT / 'tests' / 'unit'
BASELINE = Path(__file__).with_name('backend_baseline.json')
KINDS = ('name_comparisons', 'none_checks', 'module_imports_both', 'test_imports_both')
#: The backend names, known only to the backend tables and the profile type.
BACKEND_NAMES = ('slurm', 'local')
#: The only modules that may import both a SLURM module and a local one.
IMPORTS_BOTH_ALLOWED = (
    'src/tslab/study/launchers/__init__.py',
    'src/tslab/study/machine/__init__.py',
    'src/tslab/study/machine/profile.py',
    'src/tslab/study/machine/load.py',
)
#: ``.slurm is None`` and friends are allowed where a profile or a backend
#: knows about its own settings.
NONE_CHECK_EXEMPT = (
    SOURCE / 'study' / 'machine',
    SOURCE / 'study' / 'launchers' / 'slurm',
    SOURCE / 'study' / 'launchers' / 'local',
)


def _in_template(*, path: Path) -> bool:
    # why: the template files hold {{placeholders}} that do not parse
    return path.is_relative_to(TEMPLATE)


def _parse(*, path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding='utf-8'), filename=str(path))


def _string_constants(*, node: ast.AST) -> frozenset[str]:
    """The string constants anywhere under ``node``."""
    return frozenset(
        item.value
        for item in ast.walk(node)
        if isinstance(item, ast.Constant) and isinstance(item.value, str)
    )


def _names_a_backend(*, node: ast.AST) -> bool:
    """Whether a comparison operand holds a backend name or a ``launcher``."""
    has_name = bool(frozenset(BACKEND_NAMES) & _string_constants(node=node))
    has_launcher = any(
        isinstance(item, ast.Attribute) and item.attr == 'launcher' for item in ast.walk(node)
    )
    return has_name or has_launcher


def _name_comparisons(*, tree: ast.Module) -> int:
    """The comparisons against a backend name, or against ``.launcher``."""
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            if any(_names_a_backend(node=operand) for operand in (node.left, *node.comparators)):
                count += 1
        elif isinstance(node, ast.Match):
            for case in node.cases:
                if any(
                    isinstance(item, ast.MatchValue)
                    and isinstance(item.value, ast.Constant)
                    and item.value.value in BACKEND_NAMES
                    for item in ast.walk(case.pattern)
                ):
                    count += 1
    return count


def _names_a_setting(*, node: ast.AST) -> bool:
    """Whether the operand reads the ``.slurm`` or ``.local`` field of a profile or plan."""
    return any(
        isinstance(item, ast.Attribute) and item.attr in BACKEND_NAMES for item in ast.walk(node)
    )


def _none_checks(*, tree: ast.Module) -> int:
    """The ``.slurm is [not] None`` and ``.local is [not] None`` checks."""
    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        if not all(isinstance(operator, (ast.Is, ast.IsNot)) for operator in node.ops):
            continue
        operands: list[ast.expr] = [node.left, *node.comparators]
        for first, second in itertools.pairwise(operands):
            pairs = ((first, second), (second, first))
            if any(
                isinstance(other, ast.Constant)
                and other.value is None
                and _names_a_setting(node=setting)
                for setting, other in pairs
            ):
                count += 1
    return count


def _package_name(*, path: Path) -> str:
    """The package of one source or test file, for resolving relative imports."""
    roots = ((SOURCE, 'tslab'), (ROOT / 'tests', 'tests'))
    for root, prefix in roots:
        if path.is_relative_to(root):
            return '.'.join((prefix, *path.relative_to(root).parent.parts))
    message = f'{path} is not under the source or tests folder'
    raise ValueError(message)


def _imported_modules(*, tree: ast.Module, path: Path) -> frozenset[str]:
    """The absolute module paths the file imports."""
    modules: set[str] = set()
    package = _package_name(path=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module
            if node.level:
                relative_name = f'{"." * node.level}{module or ""}'
                module = resolve_name(relative_name, package)
            if module is None:
                continue
            modules.add(module)
            modules.update(f'{module}.{alias.name}' for alias in node.names)
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return frozenset(module for module in modules if module.startswith('tslab.'))


def _imports_a_side(*, modules: frozenset[str], name: str) -> bool:
    """Whether the imports touch a SLURM or a local module of ``name``."""
    prefixes = (
        f'tslab.jobs.{name}',
        f'tslab.study.launchers.{name}',
        f'tslab.study.machine.{name}',
    )
    return any(
        module == prefix or module.startswith(f'{prefix}.')
        for module in modules
        for prefix in prefixes
    )


def _imports_both_backends(*, tree: ast.Module, path: Path) -> int:
    """Whether the file imports both a SLURM module and a local one."""
    modules = _imported_modules(tree=tree, path=path)
    slurm = _imports_a_side(modules=modules, name='slurm')
    local = _imports_a_side(modules=modules, name='local')
    return int(slurm and local)


def _baseline_problems(
    *, kind: str, counts: Mapping[str, int], baseline: Mapping[str, int], hint: str
) -> list[str]:
    """Compare one kind's counts to its baseline.

    A count above the baseline is a new violation. A count below it means
    the baseline was not lowered. A baseline key with no file (a moved or
    deleted module) and a zero entry (the format omits files with 0) are
    both stale.
    """
    problems = []
    for name in sorted(set(counts) | set(baseline)):
        count = counts.get(name, 0)
        allowed = baseline.get(name, 0)
        if count > allowed:
            problems.append(f'{name}: {kind} {count} above baseline {allowed}; {hint}')
        elif count < allowed:
            problems.append(
                f'{name}: {kind} {count} below baseline {allowed}; '
                'lower the baseline in the same commit'
            )
        elif allowed == 0:
            problems.append(f'{name}: {kind} has a zero baseline entry; drop it in the same commit')
    return problems


def _source_files() -> list[Path]:
    return sorted(path for path in SOURCE.rglob('*.py') if not _in_template(path=path))


def _unit_files() -> list[Path]:
    return sorted(path for path in UNIT.rglob('*.py') if not _in_template(path=path))


def _scan(*, kind: str, files: Sequence[Path]) -> dict[str, int]:
    """The per-file counts of one kind, zero entries omitted."""
    counters = {'name_comparisons': _name_comparisons, 'none_checks': _none_checks}
    counts: dict[str, int] = {}
    for path in files:
        tree = _parse(path=path)
        if kind in ('module_imports_both', 'test_imports_both'):
            number = _imports_both_backends(tree=tree, path=path)
        else:
            number = counters[kind](tree=tree)
        if number:
            counts[path.relative_to(ROOT).as_posix()] = number
    return counts


def test_baseline_rejects_stale_and_zero_entries() -> None:
    problems = _baseline_problems(
        kind='name_comparisons',
        counts={'src/tslab/ok.py': 2, 'src/tslab/zero.py': 0},
        baseline={'src/tslab/ok.py': 2, 'src/tslab/gone.py': 3, 'src/tslab/zero.py': 0},
        hint='fix it',
    )
    assert [line.split(':', 1)[0] for line in problems] == [
        'src/tslab/gone.py',
        'src/tslab/zero.py',
    ]
    assert 'lower the baseline in the same commit' in problems[0]
    assert 'zero' in problems[1]


def test_match_cases_count_each_backend_name() -> None:
    """Each match case against a backend name counts; the pattern is its value."""
    source = (
        'match machine.launcher:\n'
        '    case "slurm":\n'
        '        pass\n'
        '    case "local":\n'
        '        pass\n'
    )
    assert _name_comparisons(tree=ast.parse(source)) == 2


def test_by_name_import_of_both_backends_counts() -> None:
    """``from tslab.jobs import slurm, local`` imports both backends."""
    tree = ast.parse('from tslab.jobs import slurm, local')
    assert _imports_both_backends(tree=tree, path=SOURCE / 'study' / 'commands.py') == 1


def test_relative_import_of_both_backends_counts(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / 'src' / 'tslab'
    module = source / 'study' / 'relative_imports.py'
    module.parent.mkdir(parents=True)
    module.write_text('from .launchers import slurm, local\n', encoding='utf-8')
    monkeypatch.setattr(sys.modules[__name__], 'ROOT', tmp_path)
    monkeypatch.setattr(sys.modules[__name__], 'SOURCE', source)

    assert _scan(kind='module_imports_both', files=[module]) == {
        'src/tslab/study/relative_imports.py': 1
    }


def test_no_backend_name_comparison() -> None:
    """No comparison against 'slurm' or 'local', and no ``.launcher`` comparison."""
    counts = _scan(kind='name_comparisons', files=_source_files())
    baseline = json.loads(BASELINE.read_text())['name_comparisons']
    problems = _baseline_problems(
        kind='name_comparisons',
        counts=counts,
        baseline=baseline,
        hint='pick the backend through BACKENDS and call a method on it',
    )
    assert problems == []


def test_no_backend_none_check() -> None:
    """No ``.slurm is None`` or ``.local is None`` outside the profile and the backends."""
    files = [
        path
        for path in _source_files()
        if not any(path.is_relative_to(exempt) for exempt in NONE_CHECK_EXEMPT)
    ]
    counts = _scan(kind='none_checks', files=files)
    baseline = json.loads(BASELINE.read_text())['none_checks']
    problems = _baseline_problems(
        kind='none_checks',
        counts=counts,
        baseline=baseline,
        hint='the backend decides about its own settings',
    )
    assert problems == []


def test_no_module_imports_both_backends() -> None:
    """One module names at most one backend, except the backend tables."""
    files = [
        path
        for path in _source_files()
        if not path.is_relative_to(SOURCE / 'jobs')
        and path.relative_to(ROOT).as_posix() not in IMPORTS_BOTH_ALLOWED
    ]
    counts = _scan(kind='module_imports_both', files=files)
    baseline = json.loads(BASELINE.read_text())['module_imports_both']
    problems = _baseline_problems(
        kind='module_imports_both',
        counts=counts,
        baseline=baseline,
        hint='move the SLURM or the local code to its backend package',
    )
    assert problems == []


def test_no_test_imports_both_backends() -> None:
    """A test file imports at most one backend."""
    counts = _scan(kind='test_imports_both', files=_unit_files())
    baseline = json.loads(BASELINE.read_text())['test_imports_both']
    problems = _baseline_problems(
        kind='test_imports_both',
        counts=counts,
        baseline=baseline,
        hint='split the test so one file imports only one backend',
    )
    assert problems == []


NEUTRAL_EXEMPT = (
    SOURCE / 'study' / 'launchers' / 'slurm',
    SOURCE / 'study' / 'launchers' / 'local',
    SOURCE / 'study' / 'machine' / '__init__.py',
    SOURCE / 'study' / 'machine' / 'profile.py',
    SOURCE / 'study' / 'machine' / 'load.py',
    SOURCE / 'study' / 'machine' / 'slurm.py',
    SOURCE / 'study' / 'slurm.py',
)
SLURM_TYPES = frozenset(('SlurmSettings', 'SlurmError', 'SlurmJobUsage'))
SLURM_MODULES = ('tslab.jobs.slurm', 'tslab.study.launchers.slurm')


def _slurm_leaks(*, tree: ast.Module, path: Path) -> list[str]:
    """The imports of a SLURM module and the uses of a SLURM type in one file."""
    modules = _imported_modules(tree=tree, path=path)
    leaks = [
        module
        for module in sorted(modules)
        if any(module == prefix or module.startswith(f'{prefix}.') for prefix in SLURM_MODULES)
    ]
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in SLURM_TYPES:
            leaks.append(node.id)
        elif isinstance(node, ast.alias) and node.name in SLURM_TYPES:
            leaks.append(node.name)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            leaks.extend(name for name in SLURM_TYPES if name in node.value.split("'"))
    return leaks


def test_neutral_study_code_names_no_slurm_type() -> None:
    """Backend-neutral study code imports no SLURM module and names no SLURM type."""
    problems = []
    for path in _source_files():
        if not path.is_relative_to(SOURCE / 'study'):
            continue
        if any(path == exempt or path.is_relative_to(exempt) for exempt in NEUTRAL_EXEMPT):
            continue
        leaks = _slurm_leaks(tree=_parse(path=path), path=path)
        if path == SOURCE / 'study' / 'launchers' / '__init__.py':
            # why: the BACKENDS table names both backends
            leaks = [leak for leak in leaks if not leak.startswith('tslab.study.launchers.slurm')]
        if leaks:
            problems.append(f'{path.relative_to(ROOT).as_posix()}: {sorted(set(leaks))}')
    assert problems == []


#: The scheduler state words that only a backend may spell.
STATE_WORDS = frozenset(
    (
        'TIMEOUT',
        'PREEMPTED',
        'NODE_FAIL',
        'OUT_OF_MEMORY',
        'PENDING',
        'RUNNING',
        'COMPLETED',
        'FAILED',
        'CANCELLED',
    )
)


def _docstring_nodes(*, tree: ast.Module) -> set[int]:
    """The ids of the string constants that are docstrings."""
    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        first = node.body[0] if node.body else None
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
            found.add(id(first.value))
    return found


def test_neutral_study_code_has_no_slurm_state_words() -> None:
    """Backend-neutral study code spells no SLURM state and says no ``SLURM`` in a string."""
    problems = []
    for path in _source_files():
        if not path.is_relative_to(SOURCE / 'study'):
            continue
        if any(path == exempt or path.is_relative_to(exempt) for exempt in NEUTRAL_EXEMPT):
            continue
        tree = _parse(path=path)
        docstrings = _docstring_nodes(tree=tree)
        words = sorted(
            {
                node.value
                for node in ast.walk(tree)
                if isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in docstrings
                and ('SLURM' in node.value or node.value in STATE_WORDS)
            }
        )
        if words:
            problems.append(f'{path.relative_to(ROOT).as_posix()}: {words}')
    assert problems == []

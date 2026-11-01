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
import itertools
import json
from pathlib import Path

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
                    isinstance(item, ast.MatchValue) and item.value in BACKEND_NAMES
                    for item in ast.walk(case.pattern)
                ):
                    count += 1
                    break
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


def _imported_modules(*, tree: ast.Module) -> frozenset[str]:
    """The absolute module paths the file imports."""
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
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


def _imports_both_backends(*, tree: ast.Module) -> int:
    """Whether the file imports both a SLURM module and a local one."""
    modules = _imported_modules(tree=tree)
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
    counters = {
        'name_comparisons': _name_comparisons,
        'none_checks': _none_checks,
        'module_imports_both': _imports_both_backends,
        'test_imports_both': _imports_both_backends,
    }
    counts: dict[str, int] = {}
    for path in files:
        number = counters[kind](tree=_parse(path=path))
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

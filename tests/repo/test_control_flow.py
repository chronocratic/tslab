"""No nested `if`, no deep function, no kind switch: a shrinking baseline.

Each count per source file is pinned in `control_flow_baseline.json`. A count
above the baseline fails (remove the construct, CODING_RULES.md Part III rules
6, 7 and 50); a count below it fails until the baseline is lowered in the same
commit, so the baseline only ever goes down.
"""

import ast
from collections.abc import Iterator, Mapping, Sequence
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'src' / 'tslab'
TEMPLATE = SOURCE / 'templates'
BASELINE = Path(__file__).with_name('control_flow_baseline.json')
KINDS = ('nested_if', 'deep_functions', 'kind_switches')


def _parse(*, source: str) -> ast.Module:
    return ast.parse(source)


def _source_files() -> list[Path]:
    return sorted(path for path in SOURCE.rglob('*.py') if TEMPLATE not in path.parents)


def _elif_chain(*, statement: ast.If) -> list[ast.If]:
    """The branches of the if/elif chain; the chain is one statement."""
    branches = [statement]
    rest = statement.orelse
    while len(rest) == 1 and isinstance(rest[0], ast.If):
        branches.append(rest[0])
        rest = rest[0].orelse
    return branches


def _branch_statements(*, branch: ast.If) -> list[ast.stmt]:
    statements = list(branch.body)
    rest = branch.orelse
    if not (len(rest) == 1 and isinstance(rest[0], ast.If)):
        statements.extend(rest)
    return statements


def _scope_subtree(node: ast.AST) -> Iterator[ast.AST]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        return  # a nested function or lambda is a new scope
    yield node
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue  # a nested function or lambda is a new scope
        yield from _scope_subtree(child)


def _scope_nodes(*, statements: Sequence[ast.stmt]) -> Iterator[ast.AST]:
    for statement in statements:
        yield from _scope_subtree(statement)


def _chain_starts(*, tree: ast.Module) -> Iterator[ast.If]:
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        parent = parents.get(node)
        if isinstance(parent, ast.If) and parent.orelse == [node]:
            continue  # an elif branch of the same statement
        yield node


def _has_nested_if(*, statement: ast.If) -> bool:
    for branch in _elif_chain(statement=statement):
        for node in _scope_nodes(statements=_branch_statements(branch=branch)):
            if isinstance(node, ast.If):
                return True
    return False


def _nested_ifs(*, tree: ast.Module) -> int:
    """Outer if statements that hold a nested if, each counted once."""
    return sum(1 for start in _chain_starts(tree=tree) if _has_nested_if(statement=start))


def _body_depth(*, branch: ast.If) -> int:
    """The block depth inside one branch; an elif branch is at the same level."""
    depth = _max_depth(statements=branch.body)
    rest = branch.orelse
    if len(rest) == 1 and isinstance(rest[0], ast.If):
        depth = max(depth, _body_depth(branch=rest[0]))
    else:
        depth = max(depth, _max_depth(statements=rest))
    return depth


def _statement_depth(*, statement: ast.stmt) -> int:
    if isinstance(statement, ast.If):
        return 1 + max(_body_depth(branch=branch) for branch in _elif_chain(statement=statement))
    if isinstance(statement, (ast.For, ast.While, ast.AsyncFor)):
        # The else of a loop is at the same level as its body (rule 7).
        return 1 + max(
            _max_depth(statements=statement.body), _max_depth(statements=statement.orelse)
        )
    if isinstance(statement, (ast.With, ast.AsyncWith)):
        return 1 + _max_depth(statements=statement.body)
    if isinstance(statement, ast.Try):
        inner = _max_depth(statements=statement.body)
        for handler in statement.handlers:
            inner = max(inner, _max_depth(statements=handler.body))
        inner = max(inner, _max_depth(statements=statement.orelse))
        inner = max(inner, _max_depth(statements=statement.finalbody))
        return 1 + inner
    if isinstance(statement, ast.Match):
        inner = 0
        for case in statement.cases:
            inner = max(inner, _max_depth(statements=case.body))
        return 1 + inner
    return 0  # a nested def or lambda is a new scope


def _max_depth(*, statements: Sequence[ast.stmt]) -> int:
    return max((_statement_depth(statement=statement) for statement in statements), default=0)


def _deep_functions(*, tree: ast.Module) -> int:
    """Functions whose block depth is more than 2."""
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and _max_depth(statements=node.body) > 2
    )


def _branch_key(*, test: ast.expr) -> tuple[str, str] | None:
    if isinstance(test, ast.Compare) and len(test.ops) == 1 and len(test.comparators) == 1:
        op = test.ops[0]
        if isinstance(op, (ast.Eq, ast.NotEq, ast.Is, ast.IsNot)):
            comparator = test.comparators[0]
            is_string = isinstance(comparator, ast.Constant) and isinstance(comparator.value, str)
            if is_string or isinstance(comparator, ast.Attribute):
                return ('compare', ast.unparse(test.left))
    if isinstance(test, ast.Call) and not test.keywords and len(test.args) == 1:
        func = test.func
        if isinstance(func, ast.Attribute) and func.attr == 'startswith':
            return ('startswith', ast.unparse(func.value))
    return None


def _is_kind_chain(*, statement: ast.If) -> bool:
    keys = [
        key
        for branch in _elif_chain(statement=statement)
        if (key := _branch_key(test=branch.test)) is not None
    ]
    return any(keys.count(key) >= 2 for key in set(keys))


def _value_cases(*, node: ast.Match) -> int:
    count = 0
    for case in node.cases:
        pattern = case.pattern
        if isinstance(pattern, ast.MatchValue):
            patterns = [pattern]
        elif isinstance(pattern, ast.MatchOr):
            patterns = pattern.patterns
        else:
            patterns = []
        if patterns and all(isinstance(sub, ast.MatchValue) for sub in patterns):
            count += 1
    return count


def _kind_switches(*, tree: ast.Module) -> int:
    """if/elif chains over one expression and match statements over values."""
    chains = sum(1 for start in _chain_starts(tree=tree) if _is_kind_chain(statement=start))
    matches = sum(
        1 for node in ast.walk(tree) if isinstance(node, ast.Match) and _value_cases(node=node) >= 2
    )
    return chains + matches


def _baseline_problems(
    *, kind: str, counts: Mapping[str, int], baseline: Mapping[str, int]
) -> list[str]:
    """Compare one kind's counts to its baseline.

    A count above the baseline is a new violation. A count below it means the
    baseline was not lowered. A baseline key with no file (a moved or deleted
    module) and a zero entry (the format omits files with 0) are both stale.
    """
    problems = []
    for name in sorted(set(counts) | set(baseline)):
        count = counts.get(name, 0)
        allowed = baseline.get(name, 0)
        if count > allowed:
            problems.append(
                f'{name}: {kind} {count} above baseline {allowed}; '
                'remove the nested if or kind switch (CODING_RULES.md Part III '
                'rules 6, 7 and 50)'
            )
        elif count < allowed:
            problems.append(
                f'{name}: {kind} {count} below baseline {allowed}; '
                'lower the baseline in the same commit'
            )
        elif name in baseline and allowed == 0:
            problems.append(
                f'{name}: {kind} baseline entry is zero; lower the baseline in the same commit'
            )
    return problems


def test_elif_chain_is_not_nesting() -> None:
    tree = _parse(source='if a:\n    x\nelif b:\n    y\nelse:\n    z\n')
    assert _nested_ifs(tree=tree) == 0


def test_if_inside_if_is_nesting() -> None:
    tree = _parse(source='if a:\n    if b:\n        x\n')
    assert _nested_ifs(tree=tree) == 1


def test_if_inside_loop_inside_if_is_nesting() -> None:
    tree = _parse(source='if a:\n    for i in x:\n        if b:\n            y\n')
    assert _nested_ifs(tree=tree) == 1


def test_nested_def_starts_a_new_scope() -> None:
    tree = _parse(
        source='def f():\n    if a:\n        def g():\n            if b:\n                x\n'
    )
    assert _nested_ifs(tree=tree) == 0


def test_two_branch_string_chain_is_a_kind_switch() -> None:
    chain = _parse(source="if k == 'a':\n    x\nelif k == 'b':\n    y\n")
    assert _kind_switches(tree=chain) == 1
    single = _parse(source="if k == 'a':\n    x\n")
    assert _kind_switches(tree=single) == 0


def test_enum_chain_is_a_kind_switch() -> None:
    tree = _parse(source='if s is State.A:\n    x\nelif s is State.B:\n    y\n')
    assert _kind_switches(tree=tree) == 1


def test_match_on_values_is_a_kind_switch() -> None:
    values = _parse(source="match k:\n    case 'a':\n        x\n    case 'b':\n        y\n")
    assert _kind_switches(tree=values) == 1
    classes = _parse(
        source='match point:\n'
        '    case Point(x=x, y=y):\n'
        '        f(x, y)\n'
        '    case Segment(a=a, b=b):\n'
        '        g(a, b)\n'
    )
    assert _kind_switches(tree=classes) == 0


def test_depth_three_function_is_deep() -> None:
    deep = _parse(
        source='def f():\n    for a in x:\n        with y:\n            if z:\n                w\n'
    )
    assert _deep_functions(tree=deep) == 1
    flat = _parse(source='def g():\n    for a in x:\n        for b in y:\n            w\n')
    assert _deep_functions(tree=flat) == 0


def test_for_else_body_counts_at_the_loop_level() -> None:
    tree = _parse(
        source='def f():\n'
        '    for a in x:\n'
        '        pass\n'
        '    else:\n'
        '        for b in y:\n'
        '            if c:\n'
        '                z\n'
    )
    assert _deep_functions(tree=tree) == 1


def test_async_for_and_async_with_count() -> None:
    tree = _parse(
        source='async def f():\n'
        '    async with y:\n'
        '        async for a in x:\n'
        '            if b:\n'
        '                z\n'
    )
    assert _deep_functions(tree=tree) == 1


def test_baseline_rejects_stale_and_zero_entries() -> None:
    problems = _baseline_problems(
        kind='nested_if',
        counts={'src/tslab/ok.py': 2, 'src/tslab/zero.py': 0},
        baseline={'src/tslab/ok.py': 2, 'src/tslab/gone.py': 3, 'src/tslab/zero.py': 0},
    )
    assert [line.split(':', 1)[0] for line in problems] == [
        'src/tslab/gone.py',
        'src/tslab/zero.py',
    ]
    assert 'lower the baseline in the same commit' in problems[0]
    assert 'zero' in problems[1]


def test_control_flow_matches_baseline() -> None:
    baseline = json.loads(BASELINE.read_text())
    counts: dict[str, dict[str, int]] = {}
    for path in _source_files():
        tree = _parse(source=path.read_text())
        counts[path.relative_to(ROOT).as_posix()] = {
            'nested_if': _nested_ifs(tree=tree),
            'deep_functions': _deep_functions(tree=tree),
            'kind_switches': _kind_switches(tree=tree),
        }
    problems = []
    for kind in KINDS:
        problems.extend(
            _baseline_problems(
                kind=kind,
                counts={name: values[kind] for name, values in counts.items()},
                baseline=baseline[kind],
            )
        )
    assert problems == []

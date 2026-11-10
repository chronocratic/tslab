"""Unit tests mirror the source, file by file.

``tests/<layer>/unit/<path>/test_<name>.py`` must name a module of
``src/tslab/<layer>/<path>/``, for ``<layer>`` in ``lab``, ``ml`` and ``ts``:
``name == module``, ``name`` starting with
``module + '_'``, or ``name == 'package'`` for the ``__init__.py`` of a
package. Helper files without a ``test_`` prefix (conftest.py, helpers.py,
fakes) are not checked. Files that do not mirror a module yet are listed in
``test_layout_baseline.json`` and must be fixed there in the same commit.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'src' / 'tslab'
LAYERS = ('lab', 'ml', 'ts')
UNIT_TREES = tuple(ROOT / 'tests' / layer / 'unit' for layer in LAYERS)
BASELINE = Path(__file__).with_name('test_layout_baseline.json')


def _module_names(*, source_dir: Path) -> set[str]:
    if not source_dir.is_dir():
        return set()
    names = set()
    for path in source_dir.iterdir():
        if path.is_file() and path.suffix == '.py' and path.name != '__init__.py':
            names.add(path.stem)
        elif path.is_dir() and (path / '__init__.py').is_file():
            names.add(path.name)
    return names


def _mirrors_a_module(*, test_file: Path) -> bool:
    layer = test_file.relative_to(ROOT / 'tests').parts[0]
    relative = test_file.relative_to(ROOT / 'tests' / layer / 'unit')
    folder = Path(layer, *relative.parts[:-1])
    name = relative.stem.removeprefix('test_')
    if name == 'package':
        return (SOURCE / folder / '__init__.py').is_file()
    return any(
        name == module or name.startswith(f'{module}_')
        for module in _module_names(source_dir=SOURCE / folder)
    )


def test_unit_tests_mirror_the_source() -> None:
    baseline = set(json.loads(BASELINE.read_text()))
    actual = {
        path.relative_to(ROOT).as_posix()
        for tree in UNIT_TREES
        for path in tree.rglob('test_*.py')
        if not _mirrors_a_module(test_file=path)
    }
    problems = [
        f'{name}: name it test_<module>.py next to the mirrored folder'
        for name in sorted(actual - baseline)
    ]
    problems.extend(
        f'{name}: remove it from the baseline in the same commit'
        for name in sorted(baseline - actual)
    )
    assert problems == []


def test_mirror_rule_examples() -> None:
    assert _mirrors_a_module(test_file=ROOT / 'tests' / 'lab' / 'unit' / 'core' / 'test_run.py')
    assert _mirrors_a_module(
        test_file=ROOT / 'tests' / 'lab' / 'unit' / 'core' / 'test_config_unions.py'
    )
    assert not _mirrors_a_module(
        test_file=ROOT / 'tests' / 'lab' / 'unit' / 'core' / 'test_nothing.py'
    )

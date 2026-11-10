"""Source text and test import rules of the lab, ml and ts layers."""

import ast
from pathlib import Path
import re

_ROOT = Path(__file__).parents[2]
LAB = _ROOT / 'src' / 'tslab' / 'lab'
_ML_LIBRARIES = re.compile(
    r'\b(torch|lightning|safetensors|sklearn|mlflow|tensorboard|chronocratic)\b', re.IGNORECASE
)
_ML_LIBRARIES_ALLOWED = (LAB / 'core' / 'seeding.py',)  # warns when torch is loaded unseeded
_WANDB = re.compile(r'\bwandb\b', re.IGNORECASE)
_WANDB_ALLOWED = (
    LAB / 'study' / 'doctor.py',
    LAB / 'study' / 'launchers' / 'slurm' / 'job_script.py',
    LAB / 'study' / 'launchers' / 'slurm' / 'followups.py',
    LAB / 'study' / 'launchers' / 'slurm' / 'backend.py',
)


def _lab_sources() -> list[Path]:
    return [p for p in sorted(LAB.rglob('*.py')) if 'templates' not in p.parts]


def _hits(*, pattern: re.Pattern[str], paths: list[Path]) -> list[str]:
    return [
        f'{p.relative_to(_ROOT)}:{n}'
        for p in paths
        for n, line in enumerate(p.read_text().splitlines(), 1)
        if pattern.search(line)
    ]


def test_lab_source_never_names_ml_libraries() -> None:
    paths = [p for p in _lab_sources() if p not in _ML_LIBRARIES_ALLOWED]
    assert _hits(pattern=_ML_LIBRARIES, paths=paths) == []


def test_lab_source_names_wandb_only_in_the_doctor_and_slurm_handoff() -> None:
    paths = [p for p in _lab_sources() if p not in _WANDB_ALLOWED]
    assert _hits(pattern=_WANDB, paths=paths) == []


def _imported(*, path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module is not None:
            names.add(node.module)
            names.update(f'{node.module}.{alias.name}' for alias in node.names)
    return names


def _violations(*, root: Path, forbidden: tuple[str, ...]) -> list[str]:
    return [
        f'{p.relative_to(_ROOT)}: {name}'
        for p in sorted(root.rglob('*.py'))
        for name in sorted(_imported(path=p))
        if any(name == f or name.startswith(f'{f}.') for f in forbidden)
    ]


def test_lab_tests_import_no_ml_or_ts_code() -> None:
    forbidden = ('torch', 'tslab.ml', 'tslab.ts', 'tests.ml', 'tests.ts')
    assert _violations(root=_ROOT / 'tests' / 'lab', forbidden=forbidden) == []


def test_ml_tests_import_no_ts_code() -> None:
    assert _violations(root=_ROOT / 'tests' / 'ml', forbidden=('tslab.ts', 'tests.ts')) == []

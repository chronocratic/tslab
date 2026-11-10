"""The dependency rules: which TSLab modules may load which libraries.

Each case imports one module in a fresh interpreter and inspects sys.modules,
so modules imported by other tests cannot hide a violation.
"""

import ast
from pathlib import Path
import subprocess
import sys

import pytest

ANALYSIS_LIBRARIES = ('pandas', 'scipy', 'matplotlib')
FORBIDDEN_IN_CORE = (
    'lightning',
    'pytorch_lightning',
    'sklearn',
    'wandb',
    'mlflow',
    'tensorboard',
    'chronocratic',
    'rich',
    'tslab.ts.adapters',
    'tslab.ml.lightning',
    'tslab.ts.tasks',
    'tslab.ts.probes',
    'tslab.lab.study',
    'tslab.lab.cli',
    'pandas',
    'scipy',
    'matplotlib',
    'tslab.lab.analysis',
)
FORBIDDEN_IN_LIGHTNING = (
    'sklearn',
    'wandb',
    'mlflow',
    'chronocratic',
    'tslab.ts.adapters',
    'tslab.ts.tasks',
    'tslab.lab.analysis',
)
FORBIDDEN_IN_PROBES = (
    'lightning',
    'pytorch_lightning',
    'wandb',
    'mlflow',
    'tensorboard',
    'chronocratic',
    'tslab.ts.adapters',
    'tslab.ml.lightning',
    'tslab.ts.tasks',
    'tslab.lab.analysis',
)
FORBIDDEN_IN_TASKS = (
    'lightning',
    'pytorch_lightning',
    'sklearn',
    'wandb',
    'mlflow',
    'tensorboard',
    'chronocratic',
    'tslab.ts.adapters',
    'tslab.ml.lightning',
    'pandas',
    'scipy',
    'matplotlib',
    'tslab.lab.analysis',
)
TRACKER_SDKS = ('wandb', 'mlflow', 'tensorboard')
FORBIDDEN_IN_TRACKING = ('lightning', 'pytorch_lightning', 'tslab.lab.analysis', *TRACKER_SDKS)
FORBIDDEN_IN_MODELS_ADAPTER = (
    'chronocratic.datasets',
    'tslab.ts.adapters.chronocratic.datasets',
    'tslab.lab.analysis',
    *TRACKER_SDKS,
)
FORBIDDEN_IN_DATASETS_ADAPTER = (
    'chronocratic.models',
    'tslab.ts.adapters.chronocratic.models',
    'tslab.lab.analysis',
    *TRACKER_SDKS,
)
FORBIDDEN_IN_STUDY = (
    'lightning',
    'pytorch_lightning',
    'sklearn',
    'chronocratic',
    'rich',
    'tslab.ts.tasks',
    'tslab.ts.adapters',
    'tslab.ml.lightning',
    'tslab.ml.tracking',
    *TRACKER_SDKS,
    'pandas',
    'scipy',
    'matplotlib',
    'tslab.lab.analysis',
)


def loaded_by(*, module: str) -> set[str]:
    code = f'import sys, {module}; print("\\n".join(sys.modules))'
    result = subprocess.run(  # noqa: S603 # why: runs this interpreter on a fixed import
        [sys.executable, '-c', code], capture_output=True, text=True, check=True
    )
    return set(result.stdout.split())


def offending(*, loaded: set[str], forbidden: tuple[str, ...]) -> list[str]:
    return sorted(
        name
        for name in loaded
        if any(name == package or name.startswith(f'{package}.') for package in forbidden)
    )


def test_package_import_loads_no_optional_layers() -> None:
    # The one fast rule: the top-level import is what every user pays for.
    assert offending(loaded=loaded_by(module='tslab'), forbidden=FORBIDDEN_IN_CORE) == []


@pytest.mark.slow
def test_lightning_package_loads_no_tracker_sdk_or_time_series_layer() -> None:
    loaded = loaded_by(module='tslab.ml.lightning')
    assert 'lightning' in loaded
    assert offending(loaded=loaded, forbidden=FORBIDDEN_IN_LIGHTNING) == []


def test_tracking_package_loads_no_lightning_or_tracker_sdk() -> None:
    # The door of tslab.ml.tracking stays importable without Lightning or a
    # tracker SDK; build_trackers imports the concrete trackers.
    loaded = loaded_by(module='tslab.ml.tracking')
    assert offending(loaded=loaded, forbidden=FORBIDDEN_IN_TRACKING) == []


@pytest.mark.slow
def test_probes_load_scikit_learn_and_nothing_else_optional() -> None:
    loaded = loaded_by(module='tslab.ts.probes')
    assert 'sklearn' in loaded
    assert offending(loaded=loaded, forbidden=FORBIDDEN_IN_PROBES) == []


@pytest.mark.slow
def test_time_series_layer_loads_neither_lightning_nor_scikit_learn() -> None:
    assert offending(loaded=loaded_by(module='tslab.ts.tasks'), forbidden=FORBIDDEN_IN_TASKS) == []


@pytest.mark.slow
@pytest.mark.chronocratic
def test_models_adapter_loads_no_dataset_code() -> None:
    loaded = loaded_by(module='tslab.ts.adapters.chronocratic.models')
    assert 'chronocratic.models' in loaded
    assert offending(loaded=loaded, forbidden=FORBIDDEN_IN_MODELS_ADAPTER) == []


@pytest.mark.slow
@pytest.mark.chronocratic
def test_datasets_adapter_loads_no_model_code() -> None:
    loaded = loaded_by(module='tslab.ts.adapters.chronocratic.datasets')
    assert 'chronocratic.datasets' in loaded
    assert offending(loaded=loaded, forbidden=FORBIDDEN_IN_DATASETS_ADAPTER) == []


@pytest.mark.slow
def test_study_loads_no_heavy_code() -> None:
    # tslab.lab.study is the import surface user study modules use; it must not
    # pull in heavy code.
    assert offending(loaded=loaded_by(module='tslab.lab.study'), forbidden=FORBIDDEN_IN_STUDY) == []


def test_chronocratic_namespace_imports_nothing() -> None:
    loaded = loaded_by(module='tslab.ts.adapters.chronocratic')
    assert (
        offending(
            loaded=loaded,
            forbidden=(
                'chronocratic.models',
                'chronocratic.datasets',
                'tslab.ts.adapters.chronocratic.models',
                'tslab.ts.adapters.chronocratic.datasets',
            ),
        )
        == []
    )


SOURCE_ROOT = Path(__file__).resolve().parents[2] / 'src' / 'tslab'
ANALYSIS_ALLOWED = (SOURCE_ROOT / 'lab' / 'analysis', SOURCE_ROOT / 'ts' / 'templates')


def _analysis_imports(*, path: Path) -> list[str]:
    found: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module is not None:
            names = [node.module]
        else:
            continue
        found.extend(
            f'{path.relative_to(SOURCE_ROOT).as_posix()}:{node.lineno} {name}'
            for name in names
            if name.split('.')[0] in ANALYSIS_LIBRARIES
        )
    return found


def test_only_analysis_imports_analysis_libraries() -> None:
    files = [
        path
        for path in SOURCE_ROOT.rglob('*.py')
        if not any(path.is_relative_to(allowed) for allowed in ANALYSIS_ALLOWED)
    ]
    assert [line for path in files for line in _analysis_imports(path=path)] == []


def test_analysis_package_loads_no_training_code() -> None:
    forbidden = (
        'lightning',
        'pytorch_lightning',
        'tslab.ml.lightning',
        'tslab.ts.adapters',
        'tslab.lab.cli',
        *TRACKER_SDKS,
    )
    assert offending(loaded=loaded_by(module='tslab.lab.analysis'), forbidden=forbidden) == []

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
CORE_MODULES = (
    'tslab',
    'tslab.core',
    'tslab.core.artifact',
    'tslab.core.config',
    'tslab.core.codec',
    'tslab.core.files',
    'tslab.core.fingerprint',
    'tslab.core.meta',
    'tslab.core.meta.instance',
    'tslab.core.meta.run',
    'tslab.core.paths',
    'tslab.core.run',
    'tslab.core.seeding',
    'tslab.core.stage',
    'tslab.core.status',
    'tslab.interfaces',
    'tslab.stages',
)
FORBIDDEN_IN_CORE = (
    'lightning',
    'pytorch_lightning',
    'sklearn',
    'wandb',
    'mlflow',
    'tensorboard',
    'chronocratic',
    'rich',
    'tslab.adapters',
    'tslab.lightning',
    'tslab.tasks',
    'tslab.probes',
    'tslab.study',
    'tslab.cli',
    'pandas',
    'scipy',
    'matplotlib',
    'tslab.analysis',
)
FORBIDDEN_IN_LIGHTNING = (
    'sklearn',
    'wandb',
    'mlflow',
    'chronocratic',
    'tslab.adapters',
    'tslab.tasks',
    'tslab.analysis',
)
FORBIDDEN_IN_PROBES = (
    'lightning',
    'pytorch_lightning',
    'wandb',
    'mlflow',
    'tensorboard',
    'chronocratic',
    'tslab.adapters',
    'tslab.lightning',
    'tslab.tasks',
    'tslab.analysis',
)
FORBIDDEN_IN_TASKS = (
    'lightning',
    'pytorch_lightning',
    'sklearn',
    'wandb',
    'mlflow',
    'tensorboard',
    'chronocratic',
    'tslab.adapters',
    'tslab.lightning',
    'pandas',
    'scipy',
    'matplotlib',
    'tslab.analysis',
)
TRACKER_SDKS = ('wandb', 'mlflow', 'tensorboard')
FORBIDDEN_IN_TRACKING = ('lightning', 'pytorch_lightning', 'tslab.analysis', *TRACKER_SDKS)
FORBIDDEN_IN_MODELS_ADAPTER = (
    'chronocratic.datasets',
    'tslab.adapters.chronocratic.datasets',
    'tslab.analysis',
    *TRACKER_SDKS,
)
FORBIDDEN_IN_DATASETS_ADAPTER = (
    'chronocratic.models',
    'tslab.adapters.chronocratic.models',
    'tslab.analysis',
    *TRACKER_SDKS,
)
FORBIDDEN_IN_STUDY = (
    'lightning',
    'pytorch_lightning',
    'sklearn',
    'chronocratic',
    'rich',
    'tslab.tasks',
    'tslab.adapters',
    'tslab.lightning',
    'tslab.tracking',
    *TRACKER_SDKS,
    'pandas',
    'scipy',
    'matplotlib',
    'tslab.analysis',
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
@pytest.mark.parametrize('module', CORE_MODULES)
def test_core_modules_load_no_optional_layers(*, module: str) -> None:
    assert offending(loaded=loaded_by(module=module), forbidden=FORBIDDEN_IN_CORE) == []


@pytest.mark.slow
def test_lightning_package_loads_no_tracker_sdk_or_time_series_layer() -> None:
    loaded = loaded_by(module='tslab.lightning')
    assert 'lightning' in loaded
    assert offending(loaded=loaded, forbidden=FORBIDDEN_IN_LIGHTNING) == []


def test_tracking_package_loads_no_lightning_or_tracker_sdk() -> None:
    # The door of tslab.tracking stays importable without Lightning or a
    # tracker SDK; build_trackers imports the concrete trackers.
    loaded = loaded_by(module='tslab.tracking')
    assert offending(loaded=loaded, forbidden=FORBIDDEN_IN_TRACKING) == []


@pytest.mark.slow
def test_probes_load_scikit_learn_and_nothing_else_optional() -> None:
    loaded = loaded_by(module='tslab.probes')
    assert 'sklearn' in loaded
    assert offending(loaded=loaded, forbidden=FORBIDDEN_IN_PROBES) == []


@pytest.mark.slow
def test_time_series_layer_loads_neither_lightning_nor_scikit_learn() -> None:
    assert offending(loaded=loaded_by(module='tslab.tasks'), forbidden=FORBIDDEN_IN_TASKS) == []


@pytest.mark.slow
@pytest.mark.chronocratic
def test_models_adapter_loads_no_dataset_code() -> None:
    loaded = loaded_by(module='tslab.adapters.chronocratic.models')
    assert 'chronocratic.models' in loaded
    assert offending(loaded=loaded, forbidden=FORBIDDEN_IN_MODELS_ADAPTER) == []


@pytest.mark.slow
@pytest.mark.chronocratic
def test_datasets_adapter_loads_no_model_code() -> None:
    loaded = loaded_by(module='tslab.adapters.chronocratic.datasets')
    assert 'chronocratic.datasets' in loaded
    assert offending(loaded=loaded, forbidden=FORBIDDEN_IN_DATASETS_ADAPTER) == []


@pytest.mark.slow
def test_study_loads_no_heavy_code() -> None:
    # tslab.study is the import surface user study modules use; it must not
    # pull in heavy code.
    assert offending(loaded=loaded_by(module='tslab.study'), forbidden=FORBIDDEN_IN_STUDY) == []


def test_chronocratic_namespace_imports_nothing() -> None:
    loaded = loaded_by(module='tslab.adapters.chronocratic')
    assert (
        offending(
            loaded=loaded,
            forbidden=(
                'chronocratic.models',
                'chronocratic.datasets',
                'tslab.adapters.chronocratic.models',
                'tslab.adapters.chronocratic.datasets',
            ),
        )
        == []
    )


SOURCE_ROOT = Path(__file__).resolve().parents[2] / 'src' / 'tslab'
ANALYSIS_ALLOWED = (SOURCE_ROOT / 'analysis', SOURCE_ROOT / 'templates')


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
        'tslab.lightning',
        'tslab.adapters',
        'tslab.cli',
        *TRACKER_SDKS,
    )
    assert offending(loaded=loaded_by(module='tslab.analysis'), forbidden=forbidden) == []

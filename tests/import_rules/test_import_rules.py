"""The dependency rules: which TSLab modules may load which libraries.

Each case imports one module in a fresh interpreter and inspects sys.modules,
so modules imported by other tests cannot hide a violation.
"""

import subprocess
import sys

import pytest

CORE_MODULES = (
    'tslab',
    'tslab.core',
    'tslab.core.artifact',
    'tslab.core.config',
    'tslab.core.fingerprint',
    'tslab.core.meta',
    'tslab.core.paths',
    'tslab.core.run',
    'tslab.core.seeding',
    'tslab.core.stage',
    'tslab.core.status',
    'tslab.interfaces',
    'tslab.stages',
    'tslab.tracking',
)
FORBIDDEN_IN_CORE = (
    'lightning',
    'pytorch_lightning',
    'sklearn',
    'wandb',
    'mlflow',
    'tensorboard',
    'chronocratic',
    'tslab.adapters',
    'tslab.lightning',
    'tslab.tasks',
)
FORBIDDEN_IN_LIGHTNING = (
    'sklearn',
    'wandb',
    'mlflow',
    'chronocratic',
    'tslab.adapters',
    'tslab.tasks',
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
)
TRACKER_SDKS = ('wandb', 'mlflow', 'tensorboard')
FORBIDDEN_IN_MODELS_ADAPTER = (
    'chronocratic.datasets',
    'tslab.adapters.chronocratic.datasets',
    *TRACKER_SDKS,
)
FORBIDDEN_IN_DATASETS_ADAPTER = (
    'chronocratic.models',
    'tslab.adapters.chronocratic.models',
    *TRACKER_SDKS,
)
FORBIDDEN_IN_STUDY = (
    'lightning',
    'pytorch_lightning',
    'sklearn',
    'chronocratic',
    'tslab.tasks',
    'tslab.adapters',
    'tslab.lightning',
    'tslab.tracking',
    *TRACKER_SDKS,
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


@pytest.mark.slow
@pytest.mark.parametrize('module', CORE_MODULES)
def test_core_modules_load_no_optional_layers(*, module: str) -> None:
    assert offending(loaded=loaded_by(module=module), forbidden=FORBIDDEN_IN_CORE) == []


@pytest.mark.slow
def test_lightning_package_loads_no_tracker_sdk_or_time_series_layer() -> None:
    loaded = loaded_by(module='tslab.lightning')
    assert 'lightning' in loaded
    assert offending(loaded=loaded, forbidden=FORBIDDEN_IN_LIGHTNING) == []


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
    # pull in heavy code. tslab.cli.study is checked by hand instead of here:
    # the CLI package __init__ imports WandbSync, which eagerly imports
    # tslab.tracking.wandb_log (a light log parser, not the wandb SDK, which
    # wandb_sync imports lazily). See the final report for that finding.
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

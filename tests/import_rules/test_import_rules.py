"""The dependency rules: which TSLab modules may load which libraries.

Each case imports one module in a fresh interpreter and inspects sys.modules,
so modules imported by other tests cannot hide a violation.
"""

import subprocess
import sys

import pytest

CORE_MODULES = (
    'tslab',
    'tslab.artifact',
    'tslab.config',
    'tslab.fingerprint',
    'tslab.interfaces',
    'tslab.meta',
    'tslab.paths',
    'tslab.run',
    'tslab.seeding',
    'tslab.stage',
    'tslab.status',
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
    'tslab.ts',
)
FORBIDDEN_IN_LIGHTNING = (
    'sklearn',
    'wandb',
    'mlflow',
    'chronocratic',
    'tslab.adapters',
    'tslab.ts',
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

from pathlib import Path
import random
import shutil
import subprocess

import numpy as np
import torch

from tslab.meta import collect_git_state, collect_versions, package_version
from tslab.seeding import seed_everything


def draw() -> tuple[float, float, float]:
    # The global generators are exactly what seed_everything must seed.
    return (
        random.random(),  # noqa: S311 # why: not cryptography; checks seeding
        float(np.random.rand()),  # noqa: NPY002 # why: the legacy global generator is under test
        float(torch.rand(1)),
    )


def test_seed_everything_repeats_all_generators() -> None:
    seed_everything(seed=3)
    first = draw()
    seed_everything(seed=3)
    assert draw() == first
    seed_everything(seed=4)
    assert draw() != first


def test_versions_include_python_and_torch() -> None:
    versions = collect_versions()
    assert versions['python'] is not None
    assert versions['torch'] is not None
    assert package_version(distribution='surely-not-installed-package') is None


def test_git_state_outside_a_repository(tmp_path: Path) -> None:
    assert collect_git_state(directory=tmp_path) == {'commit': None, 'dirty': None}


def test_git_state_inside_a_repository(tmp_path: Path) -> None:
    git = shutil.which('git')
    assert git is not None
    commands = (
        ['init', '-q'],
        ['-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-q', '--allow-empty', '-m', 'x'],
    )
    for arguments in commands:
        subprocess.run([git, *arguments], cwd=tmp_path, check=True)  # noqa: S603 # why: fixed git arguments
    clean = collect_git_state(directory=tmp_path)
    assert isinstance(clean['commit'], str)
    assert len(clean['commit']) == 40
    assert clean['dirty'] is False
    (tmp_path / 'new.txt').write_text('x', encoding='utf-8')
    assert collect_git_state(directory=tmp_path)['dirty'] is True

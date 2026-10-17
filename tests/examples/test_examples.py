"""The examples run as their docstrings say, with nothing but TSLab's public API."""

from pathlib import Path
import subprocess
import sys

import pytest

EXAMPLES = Path(__file__).parents[2] / 'examples'


def run_example(*args: str, example: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 # why: runs this repository's own example
        [sys.executable, str(EXAMPLES / example / 'main.py'), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.slow
def test_minimal_example_runs_then_reloads(tmp_path: Path) -> None:
    first = run_example(example='minimal', cwd=tmp_path)
    assert first.returncode == 0, first.stderr
    assert 'window 3: mse' in first.stdout
    second = run_example(example='minimal', cwd=tmp_path)
    assert second.stdout == first.stdout
    assert 'load smooth[w3]' in second.stderr
    third = run_example('--windows', '3', '9', '27', example='minimal', cwd=tmp_path)
    assert 'load smooth[w9]' in third.stderr
    assert 'done smooth[w27]' in third.stderr


@pytest.mark.slow
@pytest.mark.chronocratic
def test_advanced_example_skips_the_broken_attack_then_reruns_only_it(tmp_path: Path) -> None:
    one_epoch = ('--trainer.max-epochs', '1')
    first = run_example(*one_epoch, example='advanced', cwd=tmp_path)
    assert first.returncode == 0, first.stderr
    assert 'fgsm-small/h24: mse' in first.stdout
    assert 'fgsm-broken' not in first.stdout
    assert 'failed  attack[forecasting/fgsm-broken]' in first.stderr
    assert 'skipped evaluate[forecasting/fgsm-broken/h24]' in first.stderr

    fixed = run_example(*one_epoch, '--fix-broken-attack', example='advanced', cwd=tmp_path)
    assert fixed.returncode == 0, fixed.stderr
    assert 'fgsm-broken/h24: mse' in fixed.stdout
    assert 'done attack[forecasting/fgsm-broken]' in fixed.stderr
    assert 'load attack[forecasting/fgsm-small]' in fixed.stderr
    assert 'load evaluate[forecasting/clean/h24]' in fixed.stderr

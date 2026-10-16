"""M2: the toy pipeline driven from the command line, and reproduced from config.yaml."""

import json
from pathlib import Path
import subprocess
import sys

from toy_pipeline import main, read_calls, ToyConfig

import tslab

SCRIPT = Path(__file__).with_name('toy_pipeline.py')


def run_script(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 # why: runs this repository's own script
        [sys.executable, str(SCRIPT), *arguments], capture_output=True, text=True, check=False
    )


def test_help_lists_run_flags() -> None:
    result = run_script('--help')
    assert result.returncode == 0
    for flag in ('--run.force-all', '--run.rerun-from', '--run.rerun', '--run.dry-run', '--seed'):
        assert flag in result.stdout


def test_flags_drive_the_pipeline(tmp_path: Path) -> None:
    base = ('--root', str(tmp_path), '--relative-path', 'toy')
    assert run_script(*base, '--scales', '1', '3').returncode == 0
    assert read_calls(root=tmp_path) == [
        'make_data',
        'transform[x1]',
        'summarize[x1]',
        'transform[x3]',
        'summarize[x3]',
    ]
    assert run_script(*base, '--scales', '1', '3', '--run.rerun', 'transform[x3]').returncode == 0
    assert read_calls(root=tmp_path)[5:] == ['transform[x3]', 'summarize[x3]']
    dry = run_script(*base, '--scales', '1', '4', '--run.dry-run')
    assert dry.returncode == 0
    assert 'transform[x4] -> run (new)' in dry.stderr
    assert len(read_calls(root=tmp_path)) == 7


def test_invalid_flag_value_is_rejected(tmp_path: Path) -> None:
    result = run_script('--root', str(tmp_path), '--run.log-level', 'LOUD')
    assert result.returncode != 0
    assert 'LOUD' in result.stdout + result.stderr


def test_saved_config_reproduces_the_run(tmp_path: Path) -> None:
    original = ToyConfig(root=tmp_path, relative_path=Path('toy'), size=6, scales=(2.0,), seed=4)
    main(config=original)
    status_path = tmp_path / 'toy' / 'status.json'
    fingerprints = {
        instance: record['fingerprint']
        for instance, record in json.loads(status_path.read_text(encoding='utf-8'))[
            'instances'
        ].items()
    }
    reloaded = tslab.load_config(config_class=ToyConfig, path=tmp_path / 'toy' / 'config.yaml')
    assert reloaded == original
    main(config=reloaded)
    assert len(read_calls(root=tmp_path)) == 3
    reproduced = json.loads(status_path.read_text(encoding='utf-8'))['instances']
    assert {
        instance: record['fingerprint'] for instance, record in reproduced.items()
    } == fingerprints


def test_saved_config_with_overrides(tmp_path: Path) -> None:
    main(config=ToyConfig(root=tmp_path, relative_path=Path('toy')))
    loaded = tslab.load_config(config_class=ToyConfig, path=tmp_path / 'toy' / 'config.yaml')
    changed = tslab.parse_config(config_class=ToyConfig, default=loaded, args=['--seed', '1'])
    assert changed.seed == 1
    assert changed.scales == loaded.scales


def test_presets_select_and_override() -> None:
    presets = {
        'small': ('Two scales on four values', ToyConfig(size=4)),
        'large': ('Many scales on many values', ToyConfig(size=1000, scales=(1.0, 2.0, 3.0))),
    }
    config = tslab.parse_config(
        config_class=ToyConfig, presets=presets, args=['large', '--seed', '9']
    )
    assert config == ToyConfig(size=1000, scales=(1.0, 2.0, 3.0), seed=9)
    free = tslab.parse_config(
        config_class=ToyConfig, presets=presets, args=['no-preset', '--size', '3']
    )
    assert free == ToyConfig(size=3)

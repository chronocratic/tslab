"""Tests for the foreground script of a local machine."""

from pathlib import Path
import shlex

import pytest

from tslab.study.launchers.local.foreground import foreground_script


def test_foreground_script_has_no_setup(tmp_path: Path) -> None:
    lines = foreground_script(
        project_root=tmp_path, env={}, command=('python', 'x.py')
    ).splitlines()
    assert lines == [
        '#!/bin/bash',
        'set -euo pipefail',
        f'cd {shlex.quote(str(tmp_path))}',
        'export PYTHONUNBUFFERED=1',
        'exec python x.py',
    ]


def test_foreground_script_rejects_bad_env_names(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='environment variable name'):
        foreground_script(project_root=tmp_path, env={'A B': 'v'}, command=('python', 'x.py'))

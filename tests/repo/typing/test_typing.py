"""Type-checking tests: each case file states the ty error it must produce.

The first line of every file in ``cases/`` is ``# expect-error: <code>`` or
``# expect-error: none``. ty runs on each file separately, so one case can
never hide another's error.
"""

from pathlib import Path
import shutil
import subprocess

import pytest

CASES = sorted((Path(__file__).parent / 'cases').glob('*.py'))


@pytest.mark.parametrize('case', CASES, ids=[case.stem for case in CASES])
def test_type_checker_verdict(case: Path) -> None:
    ty = shutil.which('ty')
    assert ty is not None, 'ty must be installed (uv sync installs the dev group)'
    expected = (
        case.read_text(encoding='utf-8').splitlines()[0].removeprefix('# expect-error:').strip()
    )
    result = subprocess.run(  # noqa: S603 # why: runs the ty binary from this environment
        [ty, 'check', '--output-format', 'concise', str(case)],
        capture_output=True,
        text=True,
        check=False,
    )
    if expected == 'none':
        assert result.returncode == 0, result.stdout
    else:
        assert result.returncode == 1, result.stdout
        assert f'[{expected}]' in result.stdout, result.stdout

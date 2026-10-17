"""``load_study``: a ``module:attribute`` reference into a user's module."""

from collections.abc import Generator
from pathlib import Path
import sys

import pytest

from tslab.study import load_study


@pytest.fixture(autouse=True)
def _drop_loaded_studies() -> Generator[None]:
    yield
    for name in ('my_studies', 'broken_study'):
        sys.modules.pop(name, None)


def test_load_study(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / 'my_studies.py').write_text(
        'from study.configs import grid\n'
        'from tslab.study import Study\n'
        '\n'
        'rbs = Study(name="rbs", grids={"c": grid()})\n'
        'NOT_A_STUDY = 1\n'
    )
    monkeypatch.chdir(tmp_path)
    assert load_study(reference='my_studies:rbs').name == 'rbs'
    with pytest.raises(ValueError, match='reference'):
        load_study(reference='my_studies')
    with pytest.raises(ValueError, match='attribute'):
        load_study(reference='my_studies:nope')
    with pytest.raises(TypeError):
        load_study(reference='my_studies:NOT_A_STUDY')


def test_load_study_import_error_keeps_the_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / 'broken_study.py').write_text('import does_not_exist\n')
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ModuleNotFoundError, match='does_not_exist'):
        load_study(reference='broken_study:x')

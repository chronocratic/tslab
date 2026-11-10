"""``load_study``: a ``module:attribute`` reference into a user's module."""

from collections.abc import Generator
from pathlib import Path
import sys

import pytest

from tslab.study.definition import load_study


@pytest.fixture(autouse=True)
def _drop_loaded_studies() -> Generator[None]:
    yield
    for name in ('my_studies', 'broken_study'):
        sys.modules.pop(name, None)


def test_load_study(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / 'my_studies.py').write_text(
        'from tests.unit.study.configs import grid\n'
        'from tslab.study import Study\n'
        '\n'
        'rbs = Study(name="rbs", grids={"c": grid()})\n'
        'NOT_A_STUDY = 1\n'
    )
    monkeypatch.chdir(tmp_path)
    assert load_study(reference='my_studies:rbs').name == 'rbs'
    with pytest.raises(ValueError, match='reference'):
        load_study(reference='my_studies')
    with pytest.raises(ValueError, match=r"my_studies has no attribute 'nope'"):
        load_study(reference='my_studies:nope')
    with pytest.raises(ValueError, match=r"my_studies has no attribute 'study'"):
        load_study(reference='my_studies:study')
    with pytest.raises(TypeError):
        load_study(reference='my_studies:NOT_A_STUDY')


def test_load_study_import_error_keeps_the_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / 'broken_study.py').write_text('import does_not_exist\n')
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ModuleNotFoundError, match='does_not_exist'):
        load_study(reference='broken_study:x')


def test_load_study_leaves_sys_path_alone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / 'my_studies.py').write_text(
        'from tests.unit.study.configs import grid\n'
        'from tslab.study import Study\n'
        '\n'
        'rbs = Study(name="rbs", grids={"c": grid()})\n'
    )
    monkeypatch.chdir(tmp_path)
    before = list(sys.path)
    load_study(reference='my_studies:rbs')
    assert sys.path == before
    with pytest.raises(ValueError, match='attribute'):
        load_study(reference='my_studies:nope')
    assert sys.path == before


def test_load_study_sys_path_restored_on_import_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / 'broken_study.py').write_text('import does_not_exist\n')
    monkeypatch.chdir(tmp_path)
    before = list(sys.path)
    with pytest.raises(ModuleNotFoundError):
        load_study(reference='broken_study:x')
    assert sys.path == before
    assert 'broken_study' not in sys.modules


def test_load_study_prefers_the_module_of_the_current_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for folder, name in (('one', 'first'), ('two', 'second')):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / 'my_studies.py').write_text(
            'from tests.unit.study.configs import grid\n'
            'from tslab.study import Study\n'
            '\n'
            f'rbs = Study(name="{name}", grids={{"c": grid()}})\n'
        )
    monkeypatch.chdir(tmp_path / 'one')
    assert load_study(reference='my_studies:rbs').name == 'first'
    monkeypatch.chdir(tmp_path / 'two')
    assert load_study(reference='my_studies:rbs').name == 'second'


def test_load_study_prefers_the_nested_package_of_the_current_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for folder, name in (('one', 'first'), ('two', 'second')):
        root = tmp_path / folder
        study_package = root / 'pkg' / 'studies' / 'quick'
        study_package.mkdir(parents=True)
        for package in (root / 'pkg', root / 'pkg' / 'studies'):
            (package / '__init__.py').write_text('', encoding='utf-8')
        (study_package / '__init__.py').write_text(
            'from tests.unit.study.configs import grid\n'
            'from tslab.study import Study\n'
            f"study = Study(name='{name}', grids={{'g': grid()}})\n",
            encoding='utf-8',
        )

    monkeypatch.chdir(tmp_path / 'one')
    assert load_study(reference='pkg.studies.quick:study').name == 'first'
    monkeypatch.chdir(tmp_path / 'two')
    assert load_study(reference='pkg.studies.quick:study').name == 'second'


def test_three_part_reference_is_malformed() -> None:
    with pytest.raises(ValueError, match='malformed'):
        load_study(reference='a:b:c')

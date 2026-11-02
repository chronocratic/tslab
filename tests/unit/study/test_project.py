"""The project settings loader: ``tslab.toml`` at the project root."""

from pathlib import Path

import pytest

from tslab.study.project import (
    expand_study,
    load_project_settings,
    machine_for_host,
    ProjectSettings,
)


def test_missing_file_gives_defaults(tmp_path: Path) -> None:
    assert load_project_settings(start=tmp_path) == ProjectSettings(path=None)


def test_found_from_a_subfolder(tmp_path: Path) -> None:
    (tmp_path / 'tslab.toml').write_text('runs = "runs"\n', encoding='utf-8')
    start = tmp_path / 'a' / 'b'
    start.mkdir(parents=True)
    settings = load_project_settings(start=start)
    assert settings.path == tmp_path / 'tslab.toml'
    assert settings.runs == tmp_path / 'runs'


def test_nearest_file_wins(tmp_path: Path) -> None:
    (tmp_path / 'tslab.toml').write_text('studies = "outer"\n', encoding='utf-8')
    inner = tmp_path / 'inner'
    inner.mkdir()
    (inner / 'tslab.toml').write_text('studies = "inner"\n', encoding='utf-8')
    settings = load_project_settings(start=inner)
    assert settings.path == inner / 'tslab.toml'
    assert settings.studies == 'inner'


def test_unknown_key_suggests_a_close_one(tmp_path: Path) -> None:
    (tmp_path / 'tslab.toml').write_text('studys = "x"\n', encoding='utf-8')
    with pytest.raises(ValueError, match='studys') as raised:
        load_project_settings(start=tmp_path)
    assert 'studies' in str(raised.value)


@pytest.mark.parametrize(
    'text', ['studies = 3', 'runs = ""', '[machine_by_host]\n"*" = 1', 'studies = "x']
)
def test_wrong_types_name_the_file(tmp_path: Path, text: str) -> None:
    (tmp_path / 'tslab.toml').write_text(text + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match=r'tslab\.toml') as raised:
        load_project_settings(start=tmp_path)
    assert str(tmp_path / 'tslab.toml') in str(raised.value)


def test_expand_study() -> None:
    settings = ProjectSettings(path=None, studies='pkg.studies')
    assert expand_study(reference='ablation', settings=settings) == 'pkg.studies:ablation'
    assert expand_study(reference='other:x', settings=settings) == 'other:x'
    bare = ProjectSettings(path=None)
    assert expand_study(reference='ablation', settings=bare) == 'ablation'


def test_machine_for_host_keeps_file_order() -> None:
    ordered = ProjectSettings(path=None, machine_by_host=(('login*', 'slurm'), ('*', 'local')))
    assert machine_for_host(settings=ordered, host='login1') == ('login*', 'slurm')
    assert machine_for_host(settings=ordered, host='laptop') == ('*', 'local')
    catch_all_first = ProjectSettings(
        path=None, machine_by_host=(('*', 'local'), ('login*', 'slurm'))
    )
    assert machine_for_host(settings=catch_all_first, host='login1') == ('*', 'local')
    empty = ProjectSettings(path=None)
    assert machine_for_host(settings=empty, host='login1') is None

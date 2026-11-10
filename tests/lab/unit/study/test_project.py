"""The project settings loader: ``tslab.toml`` at the project root."""

from pathlib import Path

import pytest

from tslab.study.project import (
    expand_study,
    find_studies,
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
    'text',
    [
        'studies = 3',
        'runs = ""',
        '[machine_by_host]\n"*" = 1',
        '[machine_by_host]\n"*" = ""',
        'studies = "x',
    ],
)
def test_wrong_types_name_the_file(tmp_path: Path, text: str) -> None:
    (tmp_path / 'tslab.toml').write_text(text + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match=r'tslab\.toml') as raised:
        load_project_settings(start=tmp_path)
    assert str(tmp_path / 'tslab.toml') in str(raised.value)


def test_unreadable_file_is_a_value_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / 'tslab.toml').write_text('runs = "runs"\n', encoding='utf-8')

    def _unreadable(self: Path, encoding: str | None = None, errors: str | None = None) -> str:
        raise PermissionError(13, 'Permission denied', str(self))

    monkeypatch.setattr(Path, 'read_text', _unreadable)
    with pytest.raises(ValueError, match='cannot read the file') as raised:
        load_project_settings(start=tmp_path)
    assert str(tmp_path / 'tslab.toml') in str(raised.value)


def test_snapshot_keys_default_and_read(tmp_path: Path) -> None:
    (tmp_path / 'tslab.toml').write_text(
        'code_snapshot = true\ncode_snapshot_check_untracked = false\n', encoding='utf-8'
    )
    settings = load_project_settings(start=tmp_path)
    assert settings.code_snapshot is True
    assert settings.code_snapshot_check_untracked is False
    (tmp_path / 'tslab.toml').write_text('runs = "runs"\n', encoding='utf-8')
    settings = load_project_settings(start=tmp_path)
    assert settings.code_snapshot is False
    assert settings.code_snapshot_check_untracked is True


@pytest.mark.parametrize('text', ['code_snapshot = "yes"', 'code_snapshot_check_untracked = 1'])
def test_code_snapshot_not_bool_is_an_error(tmp_path: Path, text: str) -> None:
    (tmp_path / 'tslab.toml').write_text(text + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match='must be a boolean') as raised:
        load_project_settings(start=tmp_path)
    assert str(tmp_path / 'tslab.toml') in str(raised.value)


def test_expand_study() -> None:
    settings = ProjectSettings(path=None, studies='pkg.studies')
    assert expand_study(reference='ablation', settings=settings) == 'pkg.studies.ablation:study'
    assert expand_study(reference='other:x', settings=settings) == 'other:x'
    bare = ProjectSettings(path=None)
    assert expand_study(reference='ablation', settings=bare) == 'ablation'


def test_find_studies_lists_public_modules_and_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = f'project_{tmp_path.name.replace("-", "_")}'
    studies = tmp_path / package / 'studies'
    (studies / 'common').mkdir(parents=True)
    (studies / 'quick').mkdir()
    for folder in (tmp_path / package, studies, studies / 'common', studies / 'quick'):
        (folder / '__init__.py').write_text('', encoding='utf-8')
    (studies / 'full.py').write_text('', encoding='utf-8')
    (studies / '_private.py').write_text('', encoding='utf-8')
    monkeypatch.syspath_prepend(tmp_path.as_posix())

    assert find_studies(settings=ProjectSettings(path=None, studies=f'{package}.studies')) == [
        'full',
        'quick',
    ]


def test_find_studies_prefers_package_of_current_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = f'project_{tmp_path.name.replace("-", "_")}'
    for folder, name in (('one', 'first'), ('two', 'second')):
        studies = tmp_path / folder / package / 'studies'
        studies.mkdir(parents=True)
        for parent in (tmp_path / folder / package, studies):
            (parent / '__init__.py').write_text('', encoding='utf-8')
        (studies / f'{name}.py').write_text('', encoding='utf-8')

    settings = ProjectSettings(path=None, studies=f'{package}.studies')
    monkeypatch.chdir(tmp_path / 'one')
    assert find_studies(settings=settings) == ['first']
    monkeypatch.chdir(tmp_path / 'two')
    assert find_studies(settings=settings) == ['second']


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

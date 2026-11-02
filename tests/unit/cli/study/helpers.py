"""Shared helpers for the study command tests."""

from pathlib import Path
import shutil
import subprocess

import pytest

from tests.unit.study.project import write_toy_project


def toy_project(*, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str = 'project') -> Path:
    """A toy project folder, with the working directory changed into it."""
    project = write_toy_project(project=tmp_path / name)
    monkeypatch.chdir(project)
    return project


def git_project(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code_snapshot: bool = False
) -> Path:
    """The toy project as a committed git repository with a src layout and tslab.toml."""
    project = toy_project(tmp_path=tmp_path, monkeypatch=monkeypatch)
    machines = project / 'machines'
    machines.mkdir()
    (machines / 'local.toml').write_text((project / 'local.toml').read_text(encoding='utf-8'))
    (project / 'src' / 'pkg').mkdir(parents=True)
    (project / 'src' / 'pkg' / '__init__.py').write_text('# package\n', encoding='utf-8')
    toml = 'studies = "toy_studies"\n'
    if code_snapshot:
        toml += 'code_snapshot = true\n'
    toml += '\n[machine_by_host]\n"*" = "local"\n'
    (project / 'tslab.toml').write_text(toml, encoding='utf-8')
    git = shutil.which('git')
    assert git is not None
    for arguments in (('init', '-q'), ('add', '-A'), ('commit', '-q', '-m', 'init')):
        subprocess.run(  # noqa: S603 # why: fixed git command, no user input
            [git, '-c', 'user.name=t', '-c', 'user.email=t@t', *arguments],
            cwd=project,
            check=True,
            capture_output=True,
        )
    return project

"""Shared helpers for the study command tests."""

from pathlib import Path

import pytest

from tests.unit.study.project import write_toy_project


def toy_project(*, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str = 'project') -> Path:
    """A toy project folder, with the working directory changed into it."""
    project = write_toy_project(project=tmp_path / name)
    monkeypatch.chdir(project)
    return project

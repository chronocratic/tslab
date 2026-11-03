import pytest

from tslab.enums import LauncherKind


def test_launcher_kind_parses_the_profile_text() -> None:
    assert LauncherKind('slurm') is LauncherKind.SLURM
    assert LauncherKind('local') is LauncherKind.LOCAL


def test_launcher_kind_rejects_an_unknown_text() -> None:
    with pytest.raises(ValueError, match="'kubernetes' is not a valid LauncherKind"):
        LauncherKind('kubernetes')

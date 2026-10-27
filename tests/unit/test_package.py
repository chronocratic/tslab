from importlib.metadata import version
import re

import tslab


def test_version_comes_from_the_installed_package() -> None:
    assert tslab.__version__ == version('tslab')
    assert re.fullmatch(r'\d+\.\d+\.\d+.*', tslab.__version__)

"""Tests for the shell text of local job scripts."""

import pytest

from tslab.lab.jobs.local.shell import export_lines


def test_export_lines_quote_values() -> None:
    assert export_lines(env={'EXAMPLE': 'a b', 'EMPTY': ''}) == [
        "export EXAMPLE='a b'",
        "export EMPTY=''",
    ]


def test_export_lines_reject_invalid_names() -> None:
    with pytest.raises(ValueError, match='environment variable name'):
        export_lines(env={'BAD-NAME': 'value'})

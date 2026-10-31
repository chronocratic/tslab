"""Parsing and formatting of SLURM memory values."""

import pytest

from tslab.jobs.slurm import format_slurm_memory, parse_slurm_memory


def test_parse_slurm_memory_units() -> None:
    assert parse_slurm_memory(text='1234K') == 1234 * 1024
    assert parse_slurm_memory(text='1.5G') == int(1.5 * 1024**3)
    assert parse_slurm_memory(text='4000') == 4000 * 1024**2
    assert parse_slurm_memory(text='2T') == 2 * 1024**4


def test_parse_slurm_memory_empty_is_none() -> None:
    assert parse_slurm_memory(text='') is None
    assert parse_slurm_memory(text='   ') is None


def test_parse_slurm_memory_rejects_other_text() -> None:
    with pytest.raises(ValueError, match="'12X'"):
        parse_slurm_memory(text='12X')
    with pytest.raises(ValueError, match="'-1G'"):
        parse_slurm_memory(text='-1G')
    with pytest.raises(ValueError, match=r"'1\.2\.3G'"):
        parse_slurm_memory(text='1.2.3G')


def test_format_slurm_memory_rounds_up() -> None:
    assert format_slurm_memory(size=1) == '1M'
    assert format_slurm_memory(size=3 * 1024**2) == '3M'
    assert format_slurm_memory(size=3 * 1024**2 + 1) == '4M'

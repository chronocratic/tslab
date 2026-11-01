"""The ``[slurm]`` table of a machine profile."""

from pathlib import Path

import pytest

from tests.unit.study.machine.helpers import BASE_SLURM, SLURM_TABLE, write
from tslab.study.machine import load_machine, MachineProfileError, SlurmSettings


def test_slurm_time_has_the_documented_default(tmp_path: Path) -> None:
    text = 'launcher = "slurm"\noutput_root = "r"\n[slurm]\npartition = "p"\n'
    assert load_machine(path=write(tmp_path, text)).slurm == SlurmSettings(
        partition='p', time='2-00:00:00'
    )


def test_machine_profile_reads_the_new_slurm_keys(tmp_path: Path) -> None:
    text = (
        'launcher = "slurm"\noutput_root = "r"\n[slurm]\npartition = "p"\n'
        'gpu_type = "a100"\nsetup = ["module load cuda"]\nsignal_seconds = 0\n'
    )
    profile = load_machine(path=write(tmp_path, text))
    assert profile.slurm is not None
    assert (profile.slurm.gpu_type, profile.slurm.setup, profile.slurm.signal_seconds) == (
        'a100',
        ('module load cuda',),
        0,
    )


def test_machine_profile_rejects_negative_signal_seconds(tmp_path: Path) -> None:
    text = SLURM_TABLE + 'signal_seconds = -1\n'
    with pytest.raises(MachineProfileError, match='signal_seconds must be at least 0'):
        load_machine(path=write(tmp_path, BASE_SLURM + text))


def test_sbatch_extra_table_is_read_and_normalised(tmp_path: Path) -> None:
    text = (
        BASE_SLURM
        + SLURM_TABLE
        + '[slurm.sbatch_extra]\naccount = "X"\nexclusive = true\n"mail-type" = "END"\n'
    )
    profile = load_machine(path=write(tmp_path, text))
    assert profile.slurm is not None
    assert profile.slurm.sbatch_extra == (
        ('account', 'X'),
        ('exclusive', True),
        ('mail_type', 'END'),
    )


def test_sbatch_extra_list_fails_with_the_table_form(tmp_path: Path) -> None:
    text = BASE_SLURM + SLURM_TABLE + 'sbatch_extra = ["--exclude=login02", "--exclusive"]\n'
    with pytest.raises(MachineProfileError) as raised:
        load_machine(path=write(tmp_path, text))
    assert '[slurm.sbatch_extra]' in str(raised.value)
    assert 'exclude = "login02"' in str(raised.value)
    assert 'exclusive = true' in str(raised.value)


@pytest.mark.parametrize(
    ('entry', 'message'),
    [('"bad key" = "value"', 'key'), ('value = 1.5', 'string, integer, or boolean')],
)
def test_sbatch_extra_rejects_invalid_entries(tmp_path: Path, entry: str, message: str) -> None:
    text = BASE_SLURM + SLURM_TABLE + '[slurm.sbatch_extra]\n' + entry + '\n'
    with pytest.raises(MachineProfileError, match=message):
        load_machine(path=write(tmp_path, text))


@pytest.mark.parametrize(
    'line', ['gpus = -3', 'max_array_size = 0', 'max_concurrent = 0', 'cpus_per_task = 0']
)
def test_slurm_ranges(tmp_path: Path, line: str) -> None:
    text = 'launcher = "slurm"\noutput_root = "r"\n[slurm]\npartition = "p"\n' + line + '\n'
    with pytest.raises(MachineProfileError, match=line.split(' ', maxsplit=1)[0]):
        load_machine(path=write(tmp_path, text))


def test_override_slurm_ranges(tmp_path: Path) -> None:
    text = (
        'launcher = "slurm"\noutput_root = "r"\n'
        + SLURM_TABLE
        + '[[overrides]]\nmatch = { grid = "g" }\nslurm = { gpus = -1 }\n'
    )
    with pytest.raises(MachineProfileError, match=r'overrides\[0\]\.slurm\.gpus'):
        load_machine(path=write(tmp_path, text))


@pytest.mark.parametrize('field', ['partition', 'qos', 'time', 'mem_per_cpu'])
def test_slurm_strings_reject_newlines(tmp_path: Path, field: str) -> None:
    base = {'partition': '"p"', 'time': '"1:00:00"'}
    base[field] = '"a\\nrm -rf x"'
    table = ''.join(f'{key} = {value}\n' for key, value in base.items())
    text = 'launcher = "slurm"\noutput_root = "r"\n[slurm]\n' + table
    with pytest.raises(MachineProfileError, match=field):
        load_machine(path=write(tmp_path, text))


def test_sbatch_extra_table_rejects_newlines(tmp_path: Path) -> None:
    text = BASE_SLURM + SLURM_TABLE + '[slurm.sbatch_extra]\nx = "a\\nrm -rf x"\n'
    with pytest.raises(MachineProfileError, match='sbatch_extra'):
        load_machine(path=write(tmp_path, text))

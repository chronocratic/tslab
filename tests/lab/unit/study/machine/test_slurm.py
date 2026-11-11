"""The ``[slurm]`` table of a machine profile."""

from pathlib import Path

import pytest

from tests.lab.unit.study.machine.helpers import (
    BASE_SLURM,
    load_profile as load_machine,
    SLURM_TABLE,
    write,
)
from tslab.lab.study.machine.profile import MachineOverride
import tslab.lab.study.machine.slurm as slurm_machine
from tslab.lab.study.machine.slurm import SlurmSettings
from tslab.lab.study.machine.tables import MachineProfileError


def test_slurm_settings_from_normalizes_its_profile_table(tmp_path: Path) -> None:
    slurm = slurm_machine.settings_from(
        table={'slurm': {'partition': 'p', 'sbatch_extra': {'mail-type': 'END'}}},
        path=tmp_path / 'cluster.toml',
    )

    assert ('slurm', frozenset({'retry', 'chained_retry', 'wandb_sync', 'time'})) == (
        slurm_machine.TABLE,
        slurm_machine.FOLLOWUP_KEYS,
    )
    assert slurm == SlurmSettings(partition='p', sbatch_extra=(('mail_type', 'END'),))


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


@pytest.mark.parametrize('key', ['clusters', 'cluster'])
def test_sbatch_extra_rejects_clusters(tmp_path: Path, key: str) -> None:
    with pytest.raises(
        MachineProfileError, match=f'{key!r}.*multi-cluster submission is not supported'
    ):
        slurm_machine.sbatch_extra(
            value={key: 'gpu'}, location='slurm.sbatch_extra', path=tmp_path / 'm.toml'
        )


def test_override_sbatch_extra_rejects_clusters(tmp_path: Path) -> None:
    with pytest.raises(MachineProfileError, match='multi-cluster'):
        slurm_machine.slurm_overrides(
            table={'sbatch_extra': {'clusters': 'gpu'}},
            location='overrides[0].slurm',
            path=tmp_path / 'm.toml',
        )


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


def test_slurm_settings_describe_their_record_and_requested_memory() -> None:
    settings = SlurmSettings(partition='gpu', cpus_per_task=2, mem_per_cpu='2G')
    record = settings.as_record()
    assert record['partition'] == 'gpu'
    assert record['cpus_per_task'] == 2
    assert record['mem_per_cpu'] == '2G'
    assert settings.requested_memory == 2 * 2 * 1024**3
    assert SlurmSettings(partition='gpu').requested_memory is None


def test_slurm_cpus_per_run_is_cpus_per_task() -> None:
    slurm = SlurmSettings(partition='p', cpus_per_task=6)
    assert slurm.cpus == 6
    assert slurm.cpus_per_run == 6


def test_slurm_for_merges_matching_overrides_in_order() -> None:
    base = SlurmSettings(partition='p', sbatch_extra=(('account', 'X'), ('constraint', 'v100')))
    overrides = (
        MachineOverride(match={'recipe': ('a',)}, slurm={'cpus_per_task': 8}),
        MachineOverride(match={'recipe': ('b',)}, slurm={'cpus_per_task': 2}),
        MachineOverride(
            match={'recipe': ('a',)}, slurm={'sbatch_extra': (('constraint', 'a100'),)}
        ),
    )
    merged = slurm_machine.slurm_for(
        base=base, overrides=overrides, grid_name='g', values={'recipe': 'a'}
    )
    assert merged is not None
    assert merged.cpus_per_task == 8
    assert merged.sbatch_extra == (('account', 'X'), ('constraint', 'a100'))
    assert slurm_machine.slurm_for(base=None, overrides=overrides, grid_name='g', values={}) is None

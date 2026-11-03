"""The ``[local]`` table of a machine profile."""

from pathlib import Path

import pytest

from tests.unit.study.machine.helpers import (
    BASE_SLURM,
    load_profile as load_machine,
    LOCAL,
    SLURM_TABLE,
    write,
)
from tslab.study.machine import (
    FollowupSettings,
    LocalSettings,
    MachineProfileError,
    timeout_seconds,
)
import tslab.study.machine.local as local_machine


def test_local_settings_from_loads_its_profile_table(tmp_path: Path) -> None:
    local = local_machine.settings_from(
        table={'local': {'timeout': '00:05:00', 'gpus': 'auto'}}, path=tmp_path / 'local.toml'
    )

    assert ('local', frozenset({'auto_retry'})) == (
        local_machine.TABLE,
        local_machine.FOLLOWUP_KEYS,
    )
    assert local == LocalSettings(timeout='00:05:00', gpus='auto')


def test_local_profile_needs_little(tmp_path: Path) -> None:
    path = tmp_path / 'laptop.toml'
    path.write_text('launcher = "local"\noutput_root = "runs"\n')
    profile = load_machine(path=path)
    assert profile.slurm is None
    assert profile.followup == FollowupSettings()


@pytest.mark.parametrize(
    ('text', 'location'),
    [
        (LOCAL + SLURM_TABLE, 'slurm'),
        (LOCAL + '[[overrides]]\nmatch = { grid = "g" }\nslurm = { gpus = 1 }\n', 'overrides[0]'),
    ],
)
def test_local_profile_rejects_slurm_tables(tmp_path: Path, text: str, location: str) -> None:
    with pytest.raises(MachineProfileError, match=location.replace('[', '\\[').replace(']', '\\]')):
        load_machine(path=write(tmp_path, text))


def test_local_override_with_settings_only_is_fine(tmp_path: Path) -> None:
    text = LOCAL + '[[overrides]]\nmatch = { grid = "g" }\nsettings = { seed = 1 }\n'
    assert load_machine(path=write(tmp_path, text)).overrides[0].settings == {'seed': 1}


def test_local_table_defaults(tmp_path: Path) -> None:
    profile = load_machine(path=write(tmp_path, LOCAL))
    assert profile.local == LocalSettings()


def test_local_table_reads_every_key(tmp_path: Path) -> None:
    # max_parallel is left out: an explicit gpus list and the max_parallel key are an error
    text = (
        LOCAL + '[local]\n'
        'timeout = "2:00:00"\n'
        'signal_seconds = 30\n'
        'gpus = [0, 1]\n'
        'skip_busy_gpus = true\n'
        'busy_gpu_memory_fraction = 0.25\n'
        'tasks_per_gpu = 2\n'
        'keep_awake = false\n'
    )
    profile = load_machine(path=write(tmp_path, text))
    local = profile.local
    assert local is not None
    assert local == LocalSettings(
        timeout='2:00:00',
        signal_seconds=30,
        gpus=(0, 1),
        skip_busy_gpus=True,
        busy_gpu_memory_fraction=0.25,
        tasks_per_gpu=2,
        keep_awake=False,
    )
    assert timeout_seconds(local=local) == 7200.0
    assert timeout_seconds(local=LocalSettings()) is None


def test_local_gpus_auto(tmp_path: Path) -> None:
    profile = load_machine(path=write(tmp_path, LOCAL + '[local]\ngpus = "auto"\n'))
    assert profile.local is not None
    assert profile.local.gpus == 'auto'


def test_local_gpus_mps(tmp_path: Path) -> None:
    profile = load_machine(path=write(tmp_path, LOCAL + '[local]\ngpus = ["mps"]\n'))
    local = profile.local
    assert local is not None
    assert local.gpus == ('mps',)
    # the Apple GPU may be shared between tasks
    profile = load_machine(
        path=write(tmp_path, LOCAL + '[local]\ngpus = ["mps"]\ntasks_per_gpu = 2\n')
    )
    assert profile.local is not None
    assert profile.local.tasks_per_gpu == 2


def test_local_gpus_mps_with_ids_fails(tmp_path: Path) -> None:
    text = LOCAL + '[local]\ngpus = ["mps", 0]\n'
    with pytest.raises(MachineProfileError, match=r'only GPU on a Mac'):
        load_machine(path=write(tmp_path, text))


def test_local_skip_busy_with_mps_fails(tmp_path: Path) -> None:
    text = LOCAL + '[local]\ngpus = ["mps"]\nskip_busy_gpus = true\n'
    with pytest.raises(MachineProfileError, match=r'local\.skip_busy_gpus'):
        load_machine(path=write(tmp_path, text))


@pytest.mark.parametrize('fraction', ['0', '1.5'])
def test_fraction_out_of_range_fails(tmp_path: Path, fraction: str) -> None:
    text = LOCAL + f'[local]\nbusy_gpu_memory_fraction = {fraction}\n'
    with pytest.raises(
        MachineProfileError, match=r'local\.busy_gpu_memory_fraction must be above 0 and at most 1'
    ):
        load_machine(path=write(tmp_path, text))


def test_local_table_on_slurm_machine_fails(tmp_path: Path) -> None:
    text = BASE_SLURM + SLURM_TABLE + '[local]\ntimeout = "1:00"\n'
    with pytest.raises(MachineProfileError, match=r'remove the \[local\] table'):
        load_machine(path=write(tmp_path, text))


def test_override_local_fails(tmp_path: Path) -> None:
    text = (
        BASE_SLURM
        + SLURM_TABLE
        + '[[overrides]]\nmatch = { grid = "g" }\nlocal = { timeout = "1:00" }\n'
    )
    with pytest.raises(MachineProfileError, match=r'overrides\[0\]\.local'):
        load_machine(path=write(tmp_path, text))


def test_local_followup_wandb_sync_fails(tmp_path: Path) -> None:
    text = LOCAL + '[followup]\nwandb_sync = true\n'
    with pytest.raises(MachineProfileError, match=r'followup\.wandb_sync.*SLURM only'):
        load_machine(path=write(tmp_path, text))


def test_local_followup_time_fails(tmp_path: Path) -> None:
    text = LOCAL + '[followup]\ntime = "00:30:00"\n'
    with pytest.raises(MachineProfileError, match=r'followup\.time.*SLURM only'):
        load_machine(path=write(tmp_path, text))


def test_local_bad_timeout_fails(tmp_path: Path) -> None:
    text = LOCAL + '[local]\ntimeout = "soon"\n'
    with pytest.raises(MachineProfileError, match=r'local\.timeout'):
        load_machine(path=write(tmp_path, text))


def test_local_timeout_not_above_signal_seconds_fails(tmp_path: Path) -> None:
    path = write(tmp_path, LOCAL + '[local]\ntimeout = "2"\n')
    with pytest.raises(
        MachineProfileError,
        match=r'local\.signal_seconds \(120\) must be smaller than local\.timeout \(2\)',
    ):
        load_machine(path=path)


@pytest.mark.parametrize('table', ['timeout = "2"\nsignal_seconds = 0\n', 'timeout = "3"\n'])
def test_local_timeout_with_room_for_the_signal_loads(tmp_path: Path, table: str) -> None:
    load_machine(path=write(tmp_path, LOCAL + '[local]\n' + table))


@pytest.mark.parametrize('gpus', ['["0"]', '["cuda"]', '"all"'])
def test_local_gpus_not_ints_fails(tmp_path: Path, gpus: str) -> None:
    text = LOCAL + f'[local]\ngpus = {gpus}\n'
    with pytest.raises(MachineProfileError, match=r'local\.gpus'):
        load_machine(path=write(tmp_path, text))


def test_local_max_parallel_zero_fails(tmp_path: Path) -> None:
    text = LOCAL + '[local]\nmax_parallel = 0\n'
    with pytest.raises(MachineProfileError, match=r'max_parallel must be at least 1'):
        load_machine(path=write(tmp_path, text))


def test_local_gpus_with_max_parallel_fails(tmp_path: Path) -> None:
    text = LOCAL + '[local]\ngpus = [0]\nmax_parallel = 2\n'
    with pytest.raises(MachineProfileError, match=r'local\.gpus'):
        load_machine(path=write(tmp_path, text))


def test_local_tasks_per_gpu_zero_fails(tmp_path: Path) -> None:
    text = LOCAL + '[local]\ntasks_per_gpu = 0\n'
    with pytest.raises(MachineProfileError, match=r'tasks_per_gpu must be at least 1'):
        load_machine(path=write(tmp_path, text))


def test_local_tasks_per_gpu_without_gpus_fails(tmp_path: Path) -> None:
    text = LOCAL + '[local]\ntasks_per_gpu = 2\n'
    with pytest.raises(MachineProfileError, match=r'local\.tasks_per_gpu'):
        load_machine(path=write(tmp_path, text))


def test_local_skip_busy_without_gpus_fails(tmp_path: Path) -> None:
    text = LOCAL + '[local]\nskip_busy_gpus = true\n'
    with pytest.raises(MachineProfileError, match=r'local\.skip_busy_gpus'):
        load_machine(path=write(tmp_path, text))


def test_local_unknown_key_suggests(tmp_path: Path) -> None:
    text = LOCAL + '[local]\nmax_paralel = 2\n'
    with pytest.raises(MachineProfileError) as raised:
        load_machine(path=write(tmp_path, text))
    assert "did you mean 'max_parallel'?" in str(raised.value)

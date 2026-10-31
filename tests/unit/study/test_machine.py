"""``load_machine``: a machine profile, read from TOML strictly."""

from pathlib import Path

import pytest

from tslab.study.machine import (
    FollowupSettings,
    load_machine,
    LocalSettings,
    MachineProfileError,
    SlurmSettings,
    timeout_seconds,
)

CLUSTER_TOML = """\
launcher = "slurm"
output_root = "${SCRATCH}/exps"
tslab_command = ["uv", "run", "tslab"]

[settings]
"tracking.wandb_mode" = "offline"
"tracking.shell_log_path" = "{log_file}"

[env]
WANDB_ENTITY = "my-team"
UCR_ROOT = "${SCRATCH}/data/UCR"
UEA_ROOT = "${SCRATCH}/data/UEA"

[slurm]
partition = "CHANGE_ME"
qos = "normal"
time = "2-00:00:00"
cpus_per_task = 2
mem_per_cpu = "16G"
gpus = 0
max_concurrent = 100
max_array_size = 1000
node_local_tmp = true
submit_pause_seconds = 0.5

[followup]
auto_retry = false
max_attempts = 3
wandb_sync = false
time = "00:30:00"

[[overrides]]
match = { recipe = ["cost-paper"] }
slurm = { time = "14-00:00:00", qos = "long", gpus = 1 }

[[overrides]]
match = { recipe = "ts2vec-paper", dataset = ["EigenWorms", "Wallet"] }
slurm = { time = "5-00:00:00", gpus = 1, partition = "CHANGE_ME_GPU" }

[[overrides]]
match = { grid = "forecasting" }
slurm = { time = "14-00:00:00" }
settings = { "trainer.job_time_share" = 0.9 }
"""

BASE_SLURM = 'launcher = "slurm"\noutput_root = "runs"\n'


@pytest.fixture
def cluster_toml(tmp_path: Path) -> Path:
    path = tmp_path / 'cluster.toml'
    path.write_text(CLUSTER_TOML)
    return path


def test_full_example_loads(cluster_toml: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('SCRATCH', '/scratch/me')
    profile = load_machine(path=cluster_toml)
    assert profile.output_root == Path('/scratch/me/exps')
    assert profile.env['UCR_ROOT'] == '/scratch/me/data/UCR'
    assert profile.settings == {
        'tracking.wandb_mode': 'offline',
        'tracking.shell_log_path': '{log_file}',
    }
    assert profile.slurm == SlurmSettings(
        partition='CHANGE_ME',
        qos='normal',
        time='2-00:00:00',
        cpus_per_task=2,
        mem_per_cpu='16G',
        gpus=0,
    )
    assert profile.overrides[1].match == {
        'recipe': ('ts2vec-paper',),
        'dataset': ('EigenWorms', 'Wallet'),
    }
    assert profile.overrides[2].settings == {'trainer.job_time_share': 0.9}
    assert profile.tslab_command == ('uv', 'run', 'tslab')


@pytest.mark.parametrize(
    ('text', 'location'),
    [
        ('[slurm]\ncpu_per_task = 2', 'slurm.cpu_per_task'),
        ('colour = "red"', 'colour'),
        ('[slurm]\npartition = "p"\ngpus = "1"', 'slurm.gpus'),
        ('[slurm]\npartition = "p"\ngpus = true', 'slurm.gpus'),
        ('[[overrides]]\nslurm = { gpus = 1 }', 'overrides[0].match'),
        ('[[overrides]]\nmatch = { grid = "g" }\nslurm = { gpu = 1 }', 'overrides[0].slurm.gpu'),
    ],
)
def test_unknown_keys_raise(tmp_path: Path, text: str, location: str) -> None:
    base = BASE_SLURM
    if '[slurm]' not in text:
        base += '[slurm]\npartition = "p"\ntime = "1-00:00:00"\n'
    path = tmp_path / 'm.toml'
    path.write_text(base + text)
    with pytest.raises(MachineProfileError, match=location.replace('[', '\\[').replace(']', '\\]')):
        load_machine(path=path)


def test_local_profile_needs_little(tmp_path: Path) -> None:
    path = tmp_path / 'laptop.toml'
    path.write_text('launcher = "local"\noutput_root = "runs"\n')
    profile = load_machine(path=path)
    assert profile.slurm is None
    assert profile.followup == FollowupSettings()


def test_nested_settings_tables_flatten(tmp_path: Path) -> None:
    path = tmp_path / 'laptop.toml'
    path.write_text(
        'launcher = "local"\noutput_root = "runs"\n'
        '[settings]\n"trainer.max_epochs" = 3\n'
        '[settings.tracking]\nwandb_mode = "offline"\n'
    )
    assert load_machine(path=path).settings == {
        'tracking.wandb_mode': 'offline',
        'trainer.max_epochs': 3,
    }


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / 'm.toml'
    path.write_text(text)
    return path


LOCAL = 'launcher = "local"\noutput_root = "runs"\n'
SLURM_TABLE = '[slurm]\npartition = "p"\ntime = "1:00:00"\n'


@pytest.mark.parametrize(
    ('text', 'location'),
    [
        (LOCAL + SLURM_TABLE, 'slurm'),
        (LOCAL + '[[overrides]]\nmatch = { grid = "g" }\nslurm = { gpus = 1 }\n', 'overrides[0]'),
    ],
)
def test_local_profile_rejects_slurm_tables(tmp_path: Path, text: str, location: str) -> None:
    with pytest.raises(MachineProfileError, match=location.replace('[', '\\[').replace(']', '\\]')):
        load_machine(path=_write(tmp_path, text))


def test_local_override_with_settings_only_is_fine(tmp_path: Path) -> None:
    text = LOCAL + '[[overrides]]\nmatch = { grid = "g" }\nsettings = { seed = 1 }\n'
    assert load_machine(path=_write(tmp_path, text)).overrides[0].settings == {'seed': 1}


@pytest.mark.parametrize('text', ['settings = 5\n', 'settings = [1]\n'])
def test_settings_must_be_a_table(tmp_path: Path, text: str) -> None:
    with pytest.raises(MachineProfileError, match='settings must be a table'):
        load_machine(path=_write(tmp_path, LOCAL + text))


def test_override_settings_must_be_a_table(tmp_path: Path) -> None:
    text = LOCAL + '[[overrides]]\nmatch = { grid = "g" }\nsettings = 5\n'
    with pytest.raises(MachineProfileError, match=r'overrides\[0\]\.settings must be a table'):
        load_machine(path=_write(tmp_path, text))


def test_slurm_time_has_the_documented_default(tmp_path: Path) -> None:
    text = 'launcher = "slurm"\noutput_root = "r"\n[slurm]\npartition = "p"\n'
    assert load_machine(path=_write(tmp_path, text)).slurm == SlurmSettings(
        partition='p', time='2-00:00:00'
    )


def test_machine_profile_reads_the_new_slurm_keys(tmp_path: Path) -> None:
    text = (
        'launcher = "slurm"\noutput_root = "r"\n[slurm]\npartition = "p"\n'
        'gpu_type = "a100"\nsetup = ["module load cuda"]\nsignal_seconds = 0\n'
    )
    profile = load_machine(path=_write(tmp_path, text))
    assert profile.slurm is not None
    assert (profile.slurm.gpu_type, profile.slurm.setup, profile.slurm.signal_seconds) == (
        'a100',
        ('module load cuda',),
        0,
    )


def test_machine_profile_rejects_negative_signal_seconds(tmp_path: Path) -> None:
    text = SLURM_TABLE + 'signal_seconds = -1\n'
    with pytest.raises(MachineProfileError, match='signal_seconds must be at least 0'):
        load_machine(path=_write(tmp_path, BASE_SLURM + text))


def test_sbatch_extra_table_is_read_and_normalised(tmp_path: Path) -> None:
    text = (
        BASE_SLURM
        + SLURM_TABLE
        + '[slurm.sbatch_extra]\naccount = "X"\nexclusive = true\n"mail-type" = "END"\n'
    )
    profile = load_machine(path=_write(tmp_path, text))
    assert profile.slurm is not None
    assert profile.slurm.sbatch_extra == (
        ('account', 'X'),
        ('exclusive', True),
        ('mail_type', 'END'),
    )


def test_sbatch_extra_list_fails_with_the_table_form(tmp_path: Path) -> None:
    text = BASE_SLURM + SLURM_TABLE + 'sbatch_extra = ["--exclude=login02", "--exclusive"]\n'
    with pytest.raises(MachineProfileError) as raised:
        load_machine(path=_write(tmp_path, text))
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
        load_machine(path=_write(tmp_path, text))


def test_missing_file_names_the_file(tmp_path: Path) -> None:
    with pytest.raises(MachineProfileError, match=r'nope\.toml'):
        load_machine(path=tmp_path / 'nope.toml')


def test_unset_variable_in_output_root_names_the_file(tmp_path: Path) -> None:
    text = 'launcher = "local"\noutput_root = "${TSLAB_UNSET_X}/r"\n'
    with pytest.raises(MachineProfileError, match=r'm\.toml.*TSLAB_UNSET_X'):
        load_machine(path=_write(tmp_path, text))


def test_lenient_env_keeps_an_unset_variable(tmp_path: Path) -> None:
    text = LOCAL + '[env]\nROOT = "${TSLAB_UNSET_X}/d"\n'
    profile = load_machine(path=_write(tmp_path, text), lenient_env=True)
    assert profile.env == {'ROOT': '${TSLAB_UNSET_X}/d'}


@pytest.mark.parametrize(
    'line', ['gpus = -3', 'max_array_size = 0', 'max_concurrent = 0', 'cpus_per_task = 0']
)
def test_slurm_ranges(tmp_path: Path, line: str) -> None:
    text = 'launcher = "slurm"\noutput_root = "r"\n[slurm]\npartition = "p"\n' + line + '\n'
    with pytest.raises(MachineProfileError, match=line.split(' ', maxsplit=1)[0]):
        load_machine(path=_write(tmp_path, text))


def test_override_slurm_ranges(tmp_path: Path) -> None:
    text = (
        'launcher = "slurm"\noutput_root = "r"\n'
        + SLURM_TABLE
        + '[[overrides]]\nmatch = { grid = "g" }\nslurm = { gpus = -1 }\n'
    )
    with pytest.raises(MachineProfileError, match=r'overrides\[0\]\.slurm\.gpus'):
        load_machine(path=_write(tmp_path, text))


def test_empty_tslab_command_and_attempts(tmp_path: Path) -> None:
    with pytest.raises(MachineProfileError, match='tslab_command'):
        load_machine(path=_write(tmp_path, LOCAL + 'tslab_command = []\n'))
    with pytest.raises(MachineProfileError, match='max_attempts'):
        load_machine(path=_write(tmp_path, LOCAL + '[followup]\nmax_attempts = 0\n'))


def test_empty_match_list_is_rejected(tmp_path: Path) -> None:
    text = LOCAL + '[[overrides]]\nmatch = { recipe = [] }\nsettings = { seed = 1 }\n'
    with pytest.raises(MachineProfileError, match=r'match\.recipe'):
        load_machine(path=_write(tmp_path, text))


def test_settings_values_must_be_settings_values(tmp_path: Path) -> None:
    text = LOCAL + '[settings]\nwhen = 2020-01-01\n'
    with pytest.raises(MachineProfileError, match=r'm\.toml.*settings\.when'):
        load_machine(path=_write(tmp_path, text))
    text = LOCAL + '[settings]\nmany = [{ a = 1 }]\n'
    with pytest.raises(MachineProfileError, match=r'settings\.many'):
        load_machine(path=_write(tmp_path, text))


@pytest.mark.parametrize('name', ['A;rm -rf x', '1X', 'A B', 'A=B'])
def test_env_names_must_be_shell_names(tmp_path: Path, name: str) -> None:
    text = LOCAL + f'[env]\n"{name}" = "v"\n'
    with pytest.raises(MachineProfileError, match='env'):
        load_machine(path=_write(tmp_path, text))


@pytest.mark.parametrize('field', ['partition', 'qos', 'time', 'mem_per_cpu'])
def test_slurm_strings_reject_newlines(tmp_path: Path, field: str) -> None:
    base = {'partition': '"p"', 'time': '"1:00:00"'}
    base[field] = '"a\\nrm -rf x"'
    table = ''.join(f'{key} = {value}\n' for key, value in base.items())
    text = 'launcher = "slurm"\noutput_root = "r"\n[slurm]\n' + table
    with pytest.raises(MachineProfileError, match=field):
        load_machine(path=_write(tmp_path, text))


def test_sbatch_extra_table_rejects_newlines(tmp_path: Path) -> None:
    text = BASE_SLURM + SLURM_TABLE + '[slurm.sbatch_extra]\nx = "a\\nrm -rf x"\n'
    with pytest.raises(MachineProfileError, match='sbatch_extra'):
        load_machine(path=_write(tmp_path, text))


def _typo_profile(tmp_path: Path, text: str) -> Path:
    if '[' not in text:
        return _write(tmp_path, BASE_SLURM + text + '\n[slurm]\npartition = "p"\n')
    base = BASE_SLURM
    if '[slurm]' not in text:
        base += '[slurm]\npartition = "p"\ntime = "1-00:00:00"\n'
    return _write(tmp_path, base + text)


@pytest.mark.parametrize(
    ('text', 'expected'),
    [
        ('[slurm]\npartition = "p"\ngpu = 1', "slurm.gpu: unknown key; did you mean 'gpus'?"),
        ('outptu_root = "x"', "outptu_root: unknown key; did you mean 'output_root'?"),
        (
            '[followup]\nauto_retries = true',
            "followup.auto_retries: unknown key; did you mean 'auto_retry'?",
        ),
        (
            '[[overrides]]\nmatch = { grid = "g" }\nslurm = { gpu = 1 }',
            "overrides[0].slurm.gpu: unknown key; did you mean 'gpus'?",
        ),
    ],
)
def test_machine_profile_typo_suggests_the_key(tmp_path: Path, text: str, expected: str) -> None:
    with pytest.raises(MachineProfileError) as raised:
        load_machine(path=_typo_profile(tmp_path, text))
    assert expected in str(raised.value)
    assert 'Valid keys:' in str(raised.value)


def test_local_table_defaults(tmp_path: Path) -> None:
    profile = load_machine(path=_write(tmp_path, LOCAL))
    assert profile.local == LocalSettings()


def test_local_table_reads_every_key(tmp_path: Path) -> None:
    # max_parallel is left out: an explicit gpus list and the max_parallel key are an error
    text = (
        LOCAL + '[local]\n'
        'timeout = "2:00:00"\n'
        'signal_seconds = 30\n'
        'gpus = [0, 1]\n'
        'skip_busy_gpus = true\n'
        'tasks_per_gpu = 2\n'
        'keep_awake = false\n'
    )
    profile = load_machine(path=_write(tmp_path, text))
    local = profile.local
    assert local is not None
    assert local == LocalSettings(
        timeout='2:00:00',
        signal_seconds=30,
        gpus=(0, 1),
        skip_busy_gpus=True,
        tasks_per_gpu=2,
        keep_awake=False,
    )
    assert timeout_seconds(local=local) == 7200.0
    assert timeout_seconds(local=LocalSettings()) is None


def test_local_gpus_auto(tmp_path: Path) -> None:
    profile = load_machine(path=_write(tmp_path, LOCAL + '[local]\ngpus = "auto"\n'))
    assert profile.local is not None
    assert profile.local.gpus == 'auto'


def test_local_gpus_mps(tmp_path: Path) -> None:
    profile = load_machine(path=_write(tmp_path, LOCAL + '[local]\ngpus = ["mps"]\n'))
    local = profile.local
    assert local is not None
    assert local.gpus == ('mps',)
    # the Apple GPU may be shared between tasks
    profile = load_machine(
        path=_write(tmp_path, LOCAL + '[local]\ngpus = ["mps"]\ntasks_per_gpu = 2\n')
    )
    assert profile.local is not None
    assert profile.local.tasks_per_gpu == 2


def test_local_gpus_mps_with_ids_fails(tmp_path: Path) -> None:
    text = LOCAL + '[local]\ngpus = ["mps", 0]\n'
    with pytest.raises(MachineProfileError, match=r'only GPU on a Mac'):
        load_machine(path=_write(tmp_path, text))


def test_local_skip_busy_with_mps_fails(tmp_path: Path) -> None:
    text = LOCAL + '[local]\ngpus = ["mps"]\nskip_busy_gpus = true\n'
    with pytest.raises(MachineProfileError, match=r'local\.skip_busy_gpus'):
        load_machine(path=_write(tmp_path, text))


def test_local_table_on_slurm_machine_fails(tmp_path: Path) -> None:
    text = BASE_SLURM + SLURM_TABLE + '[local]\ntimeout = "1:00"\n'
    with pytest.raises(MachineProfileError, match=r'remove the \[local\] table'):
        load_machine(path=_write(tmp_path, text))


def test_override_local_fails(tmp_path: Path) -> None:
    text = (
        BASE_SLURM
        + SLURM_TABLE
        + '[[overrides]]\nmatch = { grid = "g" }\nlocal = { timeout = "1:00" }\n'
    )
    with pytest.raises(MachineProfileError, match=r'overrides\[0\]\.local'):
        load_machine(path=_write(tmp_path, text))


def test_local_followup_wandb_sync_fails(tmp_path: Path) -> None:
    text = LOCAL + '[followup]\nwandb_sync = true\n'
    with pytest.raises(MachineProfileError, match=r'followup\.wandb_sync.*SLURM only'):
        load_machine(path=_write(tmp_path, text))


def test_local_followup_time_fails(tmp_path: Path) -> None:
    text = LOCAL + '[followup]\ntime = "00:30:00"\n'
    with pytest.raises(MachineProfileError, match=r'followup\.time.*SLURM only'):
        load_machine(path=_write(tmp_path, text))


def test_local_bad_timeout_fails(tmp_path: Path) -> None:
    text = LOCAL + '[local]\ntimeout = "soon"\n'
    with pytest.raises(MachineProfileError, match=r'local\.timeout'):
        load_machine(path=_write(tmp_path, text))


@pytest.mark.parametrize('gpus', ['["0"]', '["cuda"]', '"all"'])
def test_local_gpus_not_ints_fails(tmp_path: Path, gpus: str) -> None:
    text = LOCAL + f'[local]\ngpus = {gpus}\n'
    with pytest.raises(MachineProfileError, match=r'local\.gpus'):
        load_machine(path=_write(tmp_path, text))


def test_local_max_parallel_zero_fails(tmp_path: Path) -> None:
    text = LOCAL + '[local]\nmax_parallel = 0\n'
    with pytest.raises(MachineProfileError, match=r'max_parallel must be at least 1'):
        load_machine(path=_write(tmp_path, text))


def test_local_gpus_with_max_parallel_fails(tmp_path: Path) -> None:
    text = LOCAL + '[local]\ngpus = [0]\nmax_parallel = 2\n'
    with pytest.raises(MachineProfileError, match=r'local\.gpus'):
        load_machine(path=_write(tmp_path, text))


def test_local_tasks_per_gpu_zero_fails(tmp_path: Path) -> None:
    text = LOCAL + '[local]\ntasks_per_gpu = 0\n'
    with pytest.raises(MachineProfileError, match=r'tasks_per_gpu must be at least 1'):
        load_machine(path=_write(tmp_path, text))


def test_local_tasks_per_gpu_without_gpus_fails(tmp_path: Path) -> None:
    text = LOCAL + '[local]\ntasks_per_gpu = 2\n'
    with pytest.raises(MachineProfileError, match=r'local\.tasks_per_gpu'):
        load_machine(path=_write(tmp_path, text))


def test_local_skip_busy_without_gpus_fails(tmp_path: Path) -> None:
    text = LOCAL + '[local]\nskip_busy_gpus = true\n'
    with pytest.raises(MachineProfileError, match=r'local\.skip_busy_gpus'):
        load_machine(path=_write(tmp_path, text))


def test_local_unknown_key_suggests(tmp_path: Path) -> None:
    text = LOCAL + '[local]\nmax_paralel = 2\n'
    with pytest.raises(MachineProfileError) as raised:
        load_machine(path=_write(tmp_path, text))
    assert "did you mean 'max_parallel'?" in str(raised.value)

"""``load_machine``: a machine profile, read from TOML strictly."""

from pathlib import Path

import pytest

from tslab.study.machine import FollowupSettings, load_machine, MachineProfileError, SlurmSettings

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
sbatch_extra = []
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
settings = { "trainer.max_training_time" = "4:00:00" }
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
    assert profile.overrides[2].settings == {'trainer.max_training_time': '4:00:00'}
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


def test_sbatch_extra_rejects_newlines(tmp_path: Path) -> None:
    text = 'launcher = "slurm"\noutput_root = "r"\n' + SLURM_TABLE
    text += 'sbatch_extra = ["--x=1\\nrm -rf x"]\n'
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

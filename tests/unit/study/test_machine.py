"""``load_machine``: a machine profile, read from TOML strictly."""

from pathlib import Path

import pytest

from tslab.study import FollowupSettings, load_machine, MachineProfileError, SlurmSettings

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
        ('[slurm]\ngpus = "1"', 'slurm.gpus'),
        ('[slurm]\ngpus = true', 'slurm.gpus'),
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

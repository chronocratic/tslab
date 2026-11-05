"""``load_machine``: a machine profile, read from TOML strictly."""

from collections.abc import Mapping
from pathlib import Path

import pytest

from tests.unit.study.machine.helpers import BASE_SLURM, load_profile as load_machine, LOCAL, write
from tslab.enums import LauncherKind
from tslab.study.machine import LocalSettings, MachineProfileError, SlurmSettings
import tslab.study.machine.load as machine_load
from tslab.study.machine.load import load_machine as _load_machine


def test_profile_backends_are_keyed_by_their_declared_table() -> None:
    assert {name: backend.TABLE for name, backend in machine_load.PROFILE_BACKENDS.items()} == {
        'local': 'local',
        'slurm': 'slurm',
    }


def test_load_machine_uses_selected_backend_settings_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Mapping[str, object]] = []

    class LocalProfileBackend:
        TABLE = LauncherKind.LOCAL
        FOLLOWUP_KEYS: frozenset[str] = frozenset({'auto_retry'})

        def settings_from(self, *, table: Mapping[str, object], path: Path) -> LocalSettings:
            calls.append(table)
            return LocalSettings(timeout='00:05:00')

    monkeypatch.setitem(machine_load.PROFILE_BACKENDS, LauncherKind.LOCAL, LocalProfileBackend())

    profile = _load_machine(
        path=write(tmp_path, LOCAL), environment={'HOME': str(tmp_path), 'ROOT': 'profile-root'}
    )

    assert profile.local == LocalSettings(timeout='00:05:00')
    assert calls == [{'launcher': 'local', 'output_root': 'runs'}]


def test_load_machine_expands_profile_values_from_the_given_environment(tmp_path: Path) -> None:
    path = write(
        tmp_path, 'launcher = "local"\noutput_root = "${ROOT}/runs"\n[env]\nDATA = "${ROOT}/data"\n'
    )

    profile = _load_machine(path=path, environment={'ROOT': str(tmp_path)})

    assert profile.output_root == tmp_path / 'runs'
    assert profile.env == {'DATA': f'{tmp_path}/data'}


CLUSTER_TOML = """\
launcher = "slurm"
output_root = "${SCRATCH}/exps"
tslab_command = ["uv", "run", "tslab"]

[settings]
"tracking.wandb_mode" = "offline"
"trainer.accelerator" = "gpu"

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


@pytest.fixture
def cluster_toml(tmp_path: Path) -> Path:
    path = tmp_path / 'cluster.toml'
    path.write_text(CLUSTER_TOML)
    return path


def test_full_example_loads(cluster_toml: Path) -> None:
    profile = _load_machine(path=cluster_toml, environment={'SCRATCH': '/scratch/me'})
    assert profile.output_root == Path('/scratch/me/exps')
    assert profile.env['UCR_ROOT'] == '/scratch/me/data/UCR'
    assert profile.settings == {'tracking.wandb_mode': 'offline', 'trainer.accelerator': 'gpu'}
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


def test_nested_settings_tables_flatten(tmp_path: Path) -> None:
    path = tmp_path / 'laptop.toml'
    path.write_text(
        'launcher = "local"\noutput_root = "runs"\n'
        '[settings]\n"trainer.accelerator" = "gpu"\n'
        '[settings.tracking]\nwandb_mode = "offline"\n'
    )
    assert load_machine(path=path).settings == {
        'tracking.wandb_mode': 'offline',
        'trainer.accelerator': 'gpu',
    }


@pytest.mark.parametrize('text', ['settings = 5\n', 'settings = [1]\n'])
def test_settings_must_be_a_table(tmp_path: Path, text: str) -> None:
    with pytest.raises(MachineProfileError, match='settings must be a table'):
        load_machine(path=write(tmp_path, LOCAL + text))


def test_override_settings_must_be_a_table(tmp_path: Path) -> None:
    text = LOCAL + '[[overrides]]\nmatch = { grid = "g" }\nsettings = 5\n'
    with pytest.raises(MachineProfileError, match=r'overrides\[0\]\.settings must be a table'):
        load_machine(path=write(tmp_path, text))


def test_missing_file_names_the_file(tmp_path: Path) -> None:
    with pytest.raises(MachineProfileError, match=r'nope\.toml'):
        load_machine(path=tmp_path / 'nope.toml')


def test_unset_variable_in_output_root_names_the_file(tmp_path: Path) -> None:
    text = 'launcher = "local"\noutput_root = "${TSLAB_UNSET_X}/r"\n'
    with pytest.raises(MachineProfileError, match=r'm\.toml.*TSLAB_UNSET_X'):
        load_machine(path=write(tmp_path, text))


def test_lenient_env_keeps_an_unset_variable(tmp_path: Path) -> None:
    text = LOCAL + '[env]\nROOT = "${TSLAB_UNSET_X}/d"\n'
    profile = load_machine(path=write(tmp_path, text), lenient_env=True)
    assert profile.env == {'ROOT': '${TSLAB_UNSET_X}/d'}


def test_empty_tslab_command_and_attempts(tmp_path: Path) -> None:
    with pytest.raises(MachineProfileError, match='tslab_command'):
        load_machine(path=write(tmp_path, LOCAL + 'tslab_command = []\n'))
    with pytest.raises(MachineProfileError, match='max_attempts'):
        load_machine(path=write(tmp_path, LOCAL + '[followup]\nmax_attempts = 0\n'))


def test_empty_match_list_is_rejected(tmp_path: Path) -> None:
    text = LOCAL + '[[overrides]]\nmatch = { recipe = [] }\nsettings = { seed = 1 }\n'
    with pytest.raises(MachineProfileError, match=r'match\.recipe'):
        load_machine(path=write(tmp_path, text))


def test_settings_values_must_be_settings_values(tmp_path: Path) -> None:
    text = LOCAL + '[settings]\nwhen = 2020-01-01\n'
    with pytest.raises(MachineProfileError, match=r'm\.toml.*settings\.when'):
        load_machine(path=write(tmp_path, text))
    text = LOCAL + '[settings]\nmany = [{ a = 1 }]\n'
    with pytest.raises(MachineProfileError, match=r'settings\.many'):
        load_machine(path=write(tmp_path, text))


@pytest.mark.parametrize('name', ['A;rm -rf x', '1X', 'A B', 'A=B'])
def test_env_names_must_be_shell_names(tmp_path: Path, name: str) -> None:
    text = LOCAL + f'[env]\n"{name}" = "v"\n'
    with pytest.raises(MachineProfileError, match='env'):
        load_machine(path=write(tmp_path, text))


def _typo_profile(tmp_path: Path, text: str) -> Path:
    if '[' not in text:
        return write(tmp_path, BASE_SLURM + text + '\n[slurm]\npartition = "p"\n')
    base = BASE_SLURM
    if '[slurm]' not in text:
        base += '[slurm]\npartition = "p"\ntime = "1-00:00:00"\n'
    return write(tmp_path, base + text)


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

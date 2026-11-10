"""A tiny study project written into a folder, for the study command tests.

``write_toy_project`` writes ``toy_script.py`` (a two-model config that writes
a real ``status.json`` through ``tslab.Run``), ``toy_studies.py`` (a study with
one grid over ``model`` and ``seed``) and the machine profiles ``local.toml``
and ``cluster.toml``. The command tests point ``tslab study`` at
``toy_studies:s`` from this folder, so loading, expansion, planning, launching
and state reading run together on real files.
"""

from pathlib import Path

__all__ = ['write_toy_project']

TOY_SCRIPT = '''
"""The toy experiment script of the toy study project."""

from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import Annotated

import tyro

import tslab


@dataclass(frozen=True, kw_only=True)
class Small:
    """The small model, with no settings of its own."""


@dataclass(frozen=True, kw_only=True)
class Large:
    """The large model."""

    depth: int = 9


Model = Annotated[Small, tyro.conf.subcommand(name='small')] | Annotated[
    Large, tyro.conf.subcommand(name='large')
]


@dataclass(frozen=True, kw_only=True)
class ToyConfig:
    """The config of one toy run."""

    model: Model = Small()
    seed: int = 0
    data: Path
    output_root: Path = Path('.')
    run_path: Path = Path('run')
    tracking_log: Path | None = None


def main() -> None:
    # why: the same parse config tslab.parse_config uses, so the ``model:small``
    # words of the study reach the union
    config = tyro.cli(ToyConfig, config=(tyro.conf.CascadeSubcommandArgs,))
    with tslab.Run(config=config, root=config.output_root, relative_path=config.run_path):
        time.sleep(float(os.environ.get('TOY_SLEEP', '0')))
        succeed_on = int(os.environ.get('TOY_SUCCEED_ON', '0'))
        if succeed_on:
            folder = config.output_root / config.run_path
            counter = folder / 'attempts'
            attempt = int(counter.read_text()) + 1 if counter.exists() else 1
            counter.write_text(str(attempt))
            (folder / 'last.ckpt').write_text('checkpoint')
            if attempt < succeed_on:
                raise tslab.JobTimeLimitSignal('time limit near')


if __name__ == '__main__':
    main()
'''

TOY_STUDIES = '''
"""The studies of the toy project."""

import sys

import toy_script
from tslab.study import DEFAULT_RUN_SETTINGS, Grid, Study

study = Study(
    name='s',
    grids={
        'g': Grid(
            short_name='g1',
            command=(sys.executable, 'toy_script.py'),
            config=toy_script.ToyConfig,
            axes={
                'model': {
                    'small': {'model': 'small'},
                    'large': {'model': 'large'},
                },
                'seed': {'1': {'seed': 1}, '2': {'seed': 2}},
            },
            run_settings={**DEFAULT_RUN_SETTINGS, 'data': 'data'},
        )
    },
)
'''

LOCAL_MACHINE = """\
launcher = "local"
output_root = "{output_root}"
"""

CLUSTER_MACHINE = """\
launcher = "slurm"
output_root = "{output_root}"

[slurm]
partition = "CHANGE_ME"
time = "00:10:00"
"""


def write_toy_project(*, project: Path, output_root: Path | None = None) -> Path:
    """Write the toy project files and return the project folder.

    Args:
        project: The folder to write into; created when absent.
        output_root: Where the machine profiles point at runs;
            ``<project>/runs`` when not given.
    """
    project.mkdir(parents=True, exist_ok=True)
    root = (output_root or project / 'runs').as_posix()
    (project / 'toy_script.py').write_text(TOY_SCRIPT.lstrip('\n'))
    (project / 'toy_studies.py').write_text(TOY_STUDIES.lstrip('\n'))
    (project / 'local.toml').write_text(LOCAL_MACHINE.format(output_root=root))
    (project / 'cluster.toml').write_text(CLUSTER_MACHINE.format(output_root=root))
    return project

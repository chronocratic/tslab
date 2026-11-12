# Lab reference

## Core

The names below are importable from `tslab.lab` (including `tslab.lab.REQUIRED`, the
optional marker for preset values the preset cannot know). `RunStatus` and `Tracker`
are defined in `core` and documented here.

```{eval-rst}
.. automodule:: tslab.lab
   :members:
   :imported-members:
```

## Enums

The fixed sets of choices that more than one package uses. `InstanceState`,
`LogLevel` and `RunState` are also importable from `tslab.lab` and documented under Core.

```{eval-rst}
.. automodule:: tslab.lab.enums
   :members:
   :imported-members:
   :exclude-members: InstanceState, LogLevel, RunState
```

## Studies

The study layer; user study modules import only from `tslab.lab.study`, and that
import never pulls in heavy code. The commands (`submit`, `status`,
`resubmit`, `results`) are the `tslab study` CLI; see
[Studies and HPC](../lab/studies.md).

```{eval-rst}
.. autoclass:: tslab.lab.study.Study
.. autoclass:: tslab.lab.study.Grid
.. autoclass:: tslab.lab.study.Override
.. autoclass:: tslab.lab.study.StudyRun
.. autofunction:: tslab.lab.study.setting_axis
.. autofunction:: tslab.lab.study.from_presets
.. autodata:: tslab.lab.study.DEFAULT_RUN_SETTINGS
```

## SLURM job helpers

This provisional API may move to its own package; the names stay. These general
helpers have no TSLab dependency.

```{eval-rst}
.. automodule:: tslab.lab.jobs.slurm
   :members:
   :imported-members:
```

## Command line

`tslab --help` lists the commands, and `tslab <command> --help` their flags.

| Command | What it does | Flags |
|---|---|---|
| `tslab new PATH` | copy a project template to `PATH` | `--template {lab,ts}` (default `ts` with torch installed, else `lab`), `--name` (distribution name; default: the folder name), `--layout {standalone,workspace}` (default: `workspace` inside a folder tree with a `pyproject.toml`, else `standalone`) |
| `tslab status PATH` | one line per run below `PATH`, or per instance for one run | `--stage` (only this stage), `--watch SECONDS` |
| `tslab study {submit,status,resubmit,results}` | run and follow a study; see [Studies and HPC](../lab/studies.md) | see that page |
| `tslab wandb-sync ROOT` | resync crashed, failed and killed W&B runs below `ROOT` | `--states`, `--include-offline`, `--wandb-project`, `--wandb-entity`, `--dry-run` |

`--watch SECONDS` refreshes both status commands until Ctrl-C. `tslab status`
hints point to failed-instance error files; `tslab study status` hints suggest the next action.

## Analysis

```{eval-rst}
.. automodule:: tslab.lab.analysis
   :members:
   :imported-members:
```


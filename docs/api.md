# API reference

## Core

The names below are importable from `tslab` (including `tslab.REQUIRED`, the
optional marker for preset values the preset cannot know). The interface it
re-exports (`ModelAsEncoder`) is listed under Interfaces; `RunStatus` and
`Tracker` are defined in `core` and documented here.

```{eval-rst}
.. automodule:: tslab
   :members:
   :imported-members:
   :exclude-members: ModelAsEncoder
```

## Interfaces

```{eval-rst}
.. automodule:: tslab.interfaces
   :members:
   :imported-members:
```

## Ready stages

```{eval-rst}
.. automodule:: tslab.stages
   :members:
   :imported-members:
```

## Studies

The study layer; user study modules import only from `tslab.study`, and that
import never pulls in heavy code. The commands (`submit`, `status`,
`resubmit`, `results`) are the `tslab study` CLI; see
[Studies and HPC](studies.md).

```{eval-rst}
.. autoclass:: tslab.study.Study
.. autoclass:: tslab.study.Grid
.. autoclass:: tslab.study.Override
.. autoclass:: tslab.study.StudyRun
.. autofunction:: tslab.study.setting_axis
.. autofunction:: tslab.study.from_presets
.. autodata:: tslab.study.DEFAULT_RUN_SETTINGS
```

## SLURM job helpers

This provisional API may move to its own package; the names stay. These general
helpers have no TSLab dependency.

```{eval-rst}
.. automodule:: tslab.jobs.slurm
   :members:
   :imported-members:
```

## Command line

`tslab --help` lists the commands, and `tslab <command> --help` their flags.

| Command | What it does | Flags |
|---|---|---|
| `tslab new PATH` | copy the project template to `PATH` | `--name` (distribution name; default: the folder name) |
| `tslab status PATH` | one line per run below `PATH`, or per instance for one run | `--stage` (only this stage), `--watch SECONDS` |
| `tslab study {submit,status,resubmit,results}` | run and follow a study; see [Studies and HPC](studies.md) | see that page |
| `tslab wandb-sync ROOT` | resync crashed, failed and killed W&B runs below `ROOT` | `--states`, `--include-offline`, `--wandb-project`, `--wandb-entity`, `--dry-run` |

`--watch SECONDS` refreshes both status commands until Ctrl-C. `tslab status`
hints point to failed-instance error files; `tslab study status` hints suggest the next action.

## Time series

```{eval-rst}
.. automodule:: tslab.tasks
   :members:
   :imported-members:
```

## Probes

```{eval-rst}
.. automodule:: tslab.probes
   :members:
   :imported-members:
```

## Lightning

```{eval-rst}
.. automodule:: tslab.lightning
   :members:
   :imported-members:
```

## Tracking

```{eval-rst}
.. automodule:: tslab.tracking
   :members:
   :imported-members:
```

## chronocratic adapters

```{eval-rst}
.. automodule:: tslab.adapters.chronocratic.models
   :members:
   :imported-members:

.. automodule:: tslab.adapters.chronocratic.datasets
   :members:
   :imported-members:
```

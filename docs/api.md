# API reference

## Core

The names below are importable from `tslab` (including `tslab.REQUIRED`, the
optional marker for preset values the preset cannot know). The interfaces it
re-exports (`ModelAsEncoder`, `RunStatus`, `Tracker`) are listed under
Interfaces.

```{eval-rst}
.. automodule:: tslab
   :members:
   :imported-members:
   :exclude-members: ModelAsEncoder, RunStatus, Tracker
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

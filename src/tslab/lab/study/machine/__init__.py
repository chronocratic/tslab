"""Machine profiles: one TOML file per machine, read strictly.

A profile names the launcher, the output root, settings and environment for
every run on that machine, the SLURM resources and overrides matched by axis
values. Unknown keys raise: a typo in a profile must fail at submit, not
silently change a cluster job.
"""

# why: tests/repo/test_backends.py lets no other module import both backend settings
from tslab.lab.study.machine.local import LocalSettings
from tslab.lab.study.machine.slurm import SlurmSettings

__all__ = ['LocalSettings', 'SlurmSettings']

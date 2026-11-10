"""Reporting on a study: the states, outcomes and documents of its runs.

The package holds the state collection (``state``), the exit codes and
counting of the runs (``outcome``), the log output (``report``), the results
table (``result_table``), the failure accounting (``failures``), the usage
document (``usage``), the JSON and text documents (``documents``) and the
wait for a study to settle (``wait``).

The modules import ``tslab.lab.study.context`` and ``tslab.lab.study.launchers`` at
import time, so this package does not re-export them from its ``__init__``:
it would create an import cycle. Import from a submodule,
e.g. ``from tslab.lab.study.reporting.state import StudyState``.
"""

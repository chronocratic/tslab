"""Helpers shared by the tests of the SLURM launcher."""

from functools import partial

import pytest

from tests.lab.unit.study.helpers import no_sleep
from tslab.lab.jobs.slurm import (
    query_expected_starts,
    query_job_reasons,
    query_job_start_times,
    query_job_states,
    query_job_usage,
)


def skip_query_waits(*, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the scheduler queries of the SLURM launcher retry without waiting."""
    queries = {
        'states': (query_job_states, query_expected_starts),
        'usage': (query_job_usage,),
        'backend': (query_job_start_times, query_job_reasons),
    }
    for module, functions in queries.items():
        for function in functions:
            monkeypatch.setattr(
                f'tslab.lab.study.launchers.slurm.{module}.{function.__name__}',
                partial(function, sleep=no_sleep),
            )

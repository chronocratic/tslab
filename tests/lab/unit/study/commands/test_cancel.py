"""``tslab study cancel``: cancels the selected runs and the waiting runs behind them."""

from pathlib import Path
from typing import cast, TYPE_CHECKING

import pytest

from tests.lab.unit.study.commands.test_resubmit import _chain_states, _chain_submitted
from tests.support.fake_backend import FakeBackend
from tslab.lab.enums import JobFlag, JobPhase
from tslab.lab.study.commands.cancel import cancel

if TYPE_CHECKING:
    from tslab.lab.study.context import Context


def test_cancel_upstream_cancels_waiting_downstream(
    project: Path, fake_backend: FakeBackend, caplog: pytest.LogCaptureFixture
) -> None:
    _chain_submitted(project=project, fake_backend=fake_backend)
    waiting = ('job-1_0', 'job-1_1', 'job-2_2', 'job-2_3')
    ctx = _chain_states(
        project=project,
        fake_backend=fake_backend,
        phases={
            'job-0_0': JobPhase.RUNNING,
            'job-0_1': JobPhase.RUNNING,
            **dict.fromkeys(waiting, JobPhase.QUEUED),
        },
        flags=dict.fromkeys(waiting, frozenset({JobFlag.DEPENDENCY_PENDING})),
        grids=('pretrain',),
    )

    with caplog.at_level('INFO'):
        code, _ = cancel(context=ctx, dry_run=False)

    assert code == 0
    (call,) = fake_backend.calls('cancel')
    seen = cast('Context', call['context'])
    assert {r.run.grid for r in seen.reports} == {'pretrain', 'probe'}
    assert 'also canceled 4 waiting downstream runs' in caplog.text

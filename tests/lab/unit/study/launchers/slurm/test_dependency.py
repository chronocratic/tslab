"""SLURM dependency strings for chained job targets."""

from pathlib import PurePosixPath

import pytest

from tslab.enums import AfterCondition, Satisfy
from tslab.study.launchers.slurm.dependency import dependency_text
from tslab.study.plan_after import AfterTarget


def _target(
    *,
    condition: AfterCondition = AfterCondition.OK,
    satisfy: Satisfy = Satisfy.ALL,
    delay_minutes: int = 0,
    job_tasks: tuple[str, ...] = (),
    run_paths: tuple[PurePosixPath, ...] = (),
) -> AfterTarget:
    return AfterTarget(
        condition=condition,
        delay_minutes=delay_minutes,
        satisfy=satisfy,
        job_tasks=job_tasks,
        run_paths=run_paths,
    )


@pytest.mark.parametrize(
    ('after', 'upstream', 'expected'),
    [
        ([_target(job_tasks=('41_0', '41_1'))], {}, 'afterok:41_0:41_1'),
        (
            [_target(run_paths=(PurePosixPath('pre/a'),))],
            {PurePosixPath('pre/a'): '52_3'},
            'afterok:52_3',
        ),
        (
            [
                _target(condition=AfterCondition.ANY, job_tasks=('41_0',)),
                _target(condition=AfterCondition.FAILED, job_tasks=('42_0',)),
            ],
            {},
            'afterany:41_0,afternotok:42_0',
        ),
        (
            [_target(satisfy=Satisfy.ANY, job_tasks=('41_0', '41_1'))],
            {},
            'afterok:41_0?afterok:41_1',
        ),
        (
            [_target(condition=AfterCondition.STARTED, delay_minutes=5, job_tasks=('41_0',))],
            {},
            'after:41_0+5',
        ),
        ([_target(condition=AfterCondition.STARTED, job_tasks=('41_0',))], {}, 'after:41_0'),
        ([_target()], {}, None),
    ],
)
def test_dependency_text(
    after: list[AfterTarget], upstream: dict[PurePosixPath, str], expected: str | None
) -> None:
    assert dependency_text(after=after, upstream=upstream) == expected


def test_all_dependency_rejects_a_missing_same_launch_target() -> None:
    upstream_path = PurePosixPath('pre/a')

    with pytest.raises(ValueError, match='no job id for same-launch upstream'):
        dependency_text(after=[_target(run_paths=(upstream_path,))], upstream={})

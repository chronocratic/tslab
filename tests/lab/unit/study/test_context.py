"""Study context selection keeps every chained ancestor available."""

import dataclasses
from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
from types import MappingProxyType

import pytest

from tslab.lab.core.status import InstanceRecord, RunStatusSnapshot, utc_now_iso, write_status
from tslab.lab.enums import InstanceState, RunState, StudyState
from tslab.lab.study import After, Grid, Study
from tslab.lab.study.context import _upstream_paths, context_for, Selection
from tslab.lab.study.definition import expand_runs, StudyRun


@dataclass(frozen=True, kw_only=True)
class _ChainConfig:
    output_root: Path = Path('runs')
    run_path: Path = Path('run')


def _deep_chain() -> tuple[Study, list[StudyRun]]:
    grids = {
        name: Grid(
            short_name=name,
            command=('python', '-m', name),
            config=_ChainConfig,
            axes={'dataset': {'a': {}}},
            after=() if upstream is None else (After(grid=upstream),),
            run_path='{grid}/{dataset}',
        )
        for name, upstream in (('d', 'c'), ('b', 'a'), ('a', None), ('c', 'b'))
    }
    study = Study(name='deep', grids=grids)
    return study, expand_runs(study=study)


def test_selected_middle_and_leaf_grids_collect_transitive_ancestors() -> None:
    study, runs = _deep_chain()
    selected = [run for run in runs if run.grid in {'c', 'd'}]

    upstream = _upstream_paths(study=study, runs=selected)

    assert upstream == {PurePosixPath('a/a'), PurePosixPath('b/a'), PurePosixPath('c/a')}


@pytest.mark.parametrize(
    ('selected_grid', 'upstream_grids'), [('c', {'a', 'b'}), ('d', {'a', 'b', 'c'})]
)
def test_grid_selection_derives_failures_from_unselected_ancestors(
    tmp_path: Path, selected_grid: str, upstream_grids: set[str]
) -> None:
    study, _ = _deep_chain()
    output_root = tmp_path / 'runs'
    machine_path = tmp_path / 'machine.toml'
    machine_path.write_text(
        f'launcher = "local"\noutput_root = {json.dumps(output_root.as_posix())}\n',
        encoding='utf-8',
    )
    write_status(
        path=output_root / 'deep' / 'a' / 'a' / 'status.json',
        snapshot=RunStatusSnapshot(
            run_state=RunState.FAILED,
            updated_at=utc_now_iso(),
            instances={'train': InstanceRecord(state=InstanceState.FAILED)},
        ),
    )

    context = context_for(
        study=study,
        machine_path=machine_path,
        selection=Selection(grids=(selected_grid,)),
        project_root=tmp_path,
        local_cpus=1,
        environment={},
    )

    assert isinstance(context.resolved, MappingProxyType)
    assert isinstance(context.upstream_reports, MappingProxyType)
    resolved = dict(context.resolved)
    upstream_reports = dict(context.upstream_reports)
    snapshot = dataclasses.replace(context, resolved=resolved, upstream_reports=upstream_reports)
    resolved.clear()
    upstream_reports.clear()
    assert snapshot.resolved
    assert snapshot.upstream_reports
    resolved_backing = dict(context.resolved)
    wrapped = dataclasses.replace(context, resolved=MappingProxyType(resolved_backing))
    resolved_backing.clear()
    assert wrapped.resolved
    assert {report.run.grid for report in context.reports} == {selected_grid}
    assert {report.run.grid for report in context.upstream_reports.values()} == upstream_grids
    assert {report.state for report in context.reports} == {StudyState.DEPENDENCY_FAILED}
    assert all(
        report.state is StudyState.DEPENDENCY_FAILED
        for report in context.upstream_reports.values()
        if report.run.grid != 'a'
    )

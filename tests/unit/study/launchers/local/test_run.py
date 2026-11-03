"""Local study launch tests."""

from collections.abc import Mapping, Sequence
import contextlib
import functools
from io import StringIO
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import ClassVar, TextIO

import pytest

from tests.unit.study.configs import grid
from tests.unit.study.helpers import study_directory
from tslab.study import setting_axis, Study
from tslab.study.definition import expand_runs
from tslab.study.launchers import launch as _launch, LaunchReport, OnTaskState
from tslab.study.launchers.local import run as _local_run
from tslab.study.launchers.local.pid_file import LAUNCH_FILE
from tslab.study.launchers.local.run import (
    local_job_id,
    LocalLauncher as _LocalLauncher,
    write_job_files,
)
from tslab.study.machine import LocalSettings, MachineProfile
from tslab.study.paths import study_folder
from tslab.study.plan import assign_indices, JobGroup, plan_jobs, resolve_runs
from tslab.study.records import read_jobs


class LocalLauncher(_LocalLauncher):
    """A local launcher test adapter with fixed process metadata."""

    def __init__(  # noqa: PLR0913 # why: test adapter mirrors the launcher constructor inputs
        self,
        *,
        project_root: Path,
        env: Mapping[str, str],
        local: LocalSettings,
        gpu_ids: tuple[str, ...] = (),
        stream: TextIO | None = None,
        on_task_state: OnTaskState | None = None,
        grace_seconds: float = 30.0,
    ) -> None:
        super().__init__(
            project_root=project_root,
            env=env,
            local=local,
            pid=123,
            host='test-host',
            now=lambda: '2026-01-02T03:04:05+00:00',
            gpu_ids=gpu_ids,
            stream=stream,
            on_task_state=on_task_state,
            grace_seconds=grace_seconds,
        )


@pytest.fixture
def machine(tmp_path: Path) -> MachineProfile:
    return MachineProfile(
        path=tmp_path / 'machine.toml',
        launcher='local',
        output_root=tmp_path / 'out',
        local=LocalSettings(),
    )


@pytest.fixture
def study() -> Study:
    return Study(name='s', grids={'g': grid(short_name='g1')})


def groups_of(study: Study, machine: MachineProfile) -> list[JobGroup]:
    runs = expand_runs(study=study)
    resolved = resolve_runs(study=study, machine=machine, runs=runs, local_cpus=1)
    return plan_jobs(
        study=study,
        machine=machine,
        resolved=resolved,
        indices=assign_indices(runs=runs, jobs=[]),
        extra_args={},
        attempts={},
        first_job_file=3,
        environment={},
    )


def test_local_job_id_uses_given_process_id() -> None:
    assert local_job_id(job_file=4, pid=123) == 'local-123-4'


def launch(  # noqa: PLR0913 # why: test adapter mirrors backend launch inputs
    *,
    launcher: LocalLauncher,
    machine: MachineProfile,
    study: Study,
    groups: list[JobGroup],
    project_root: Path,
    dry_run: bool,
    start_delay_seconds: int | None = None,
) -> LaunchReport:
    return _launch(
        launcher=launcher,
        write_job_files=write_job_files,
        machine=machine,
        study=study,
        groups=groups,
        project_root=project_root,
        dry_run=dry_run,
        start_delay_seconds=start_delay_seconds,
        now=lambda: '2026-01-02T03:04:05+00:00',
    )


def test_local_launcher_runs_in_order_and_continues_after_failure(
    machine: MachineProfile, tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    machine = MachineProfile(
        path=machine.path,
        launcher='local',
        output_root=machine.output_root,
        local=machine.local or LocalSettings(),
    )
    script = tmp_path / 'run.py'
    script.write_text(
        'import sys\n'
        'args = sys.argv[1:]\n'
        "print(' '.join(args))\n"
        "if '--run-path' in args and args[args.index('--run-path') + 1].endswith('/7'):\n"
        '    sys.exit(1)\n'
    )
    study = Study(
        name='s', grids={'g': grid(short_name='g1', command=(sys.executable, str(script)))}
    )
    runs = expand_runs(study=study)
    resolved = resolve_runs(study=study, machine=machine, runs=runs, local_cpus=1)
    groups = plan_jobs(
        study=study,
        machine=machine,
        resolved=resolved,
        indices=assign_indices(runs=runs, jobs=[]),
        extra_args={},
        attempts={},
        first_job_file=1,
        environment={},
    )
    assert len(groups) == 1
    report = launch(
        launcher=LocalLauncher(project_root=tmp_path, env={}, local=LocalSettings()),
        machine=machine,
        study=study,
        groups=groups,
        project_root=tmp_path,
        dry_run=False,
    )
    assert report.submitted[0].job_id.startswith('local-')
    assert len(report.submitted[0].tasks) == 4
    folder = study_folder(machine=machine, study=study)
    log = folder / 'logs' / 'g1_2.out'
    assert log.read_text().startswith('=== tslab attempt 1 | job local-')
    console = capfd.readouterr().out
    assert '--run-path' in console
    # launching the same tasks again (attempt 2) appends: the log holds both headers
    attempts = {run.run_path: 1 for run in runs}
    groups2 = plan_jobs(
        study=study,
        machine=machine,
        resolved=resolved,
        indices=assign_indices(runs=runs, jobs=[]),
        extra_args={},
        attempts=attempts,
        first_job_file=2,
        environment={},
    )
    launch(
        launcher=LocalLauncher(project_root=tmp_path, env={}, local=LocalSettings()),
        machine=machine,
        study=study,
        groups=groups2,
        project_root=tmp_path,
        dry_run=False,
    )
    assert log.read_text().count('=== tslab attempt') == 2


def test_local_launcher_writes_to_given_stream(machine: MachineProfile, tmp_path: Path) -> None:
    local_machine = MachineProfile(
        path=machine.path, launcher='local', output_root=machine.output_root, slurm=None
    )
    script = tmp_path / 'echo.py'
    script.write_text("print('TASK-OUTPUT')")
    study = Study(
        name='s', grids={'g': grid(short_name='g1', command=(sys.executable, str(script)))}
    )
    stream = StringIO()
    launch(
        launcher=LocalLauncher(project_root=tmp_path, env={}, local=LocalSettings(), stream=stream),
        machine=local_machine,
        study=study,
        groups=groups_of(study, local_machine),
        project_root=tmp_path,
        dry_run=False,
    )
    assert 'TASK-OUTPUT' in stream.getvalue()


def test_detached_local_launcher_keeps_task_lines_out_of_the_console(
    machine: MachineProfile, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('TSLAB_DETACHED', '1')
    local_machine = MachineProfile(
        path=machine.path, launcher='local', output_root=machine.output_root, slurm=None
    )
    script = tmp_path / 'echo.py'
    script.write_text("print('TASK-OUTPUT')")
    study = Study(
        name='s', grids={'g': grid(short_name='g1', command=(sys.executable, str(script)))}
    )
    stream = StringIO()
    launch(
        launcher=LocalLauncher(project_root=tmp_path, env={}, local=LocalSettings(), stream=stream),
        machine=local_machine,
        study=study,
        groups=groups_of(study, local_machine),
        project_root=tmp_path,
        dry_run=False,
    )
    assert 'TASK-OUTPUT' not in stream.getvalue()
    assert '=== task 0 started' in stream.getvalue()


def test_local_launch_records_the_job_before_tasks_run_and_survives_a_missing_command(
    machine: MachineProfile, tmp_path: Path
) -> None:
    study = Study(
        name='s', grids={'g': grid(short_name='g1', command=(str(tmp_path / 'no-such-command'),))}
    )
    runs = expand_runs(study=study)
    groups = plan_jobs(
        study=study,
        machine=machine,
        resolved=resolve_runs(study=study, machine=machine, runs=runs, local_cpus=1),
        indices=assign_indices(runs=runs, jobs=[]),
        extra_args={},
        attempts={},
        first_job_file=1,
        environment={},
    )
    report = launch(
        launcher=LocalLauncher(project_root=tmp_path, env={}, local=LocalSettings()),
        machine=machine,
        study=study,
        groups=groups,
        project_root=tmp_path,
        dry_run=False,
    )
    folder = study_folder(machine=machine, study=study)
    assert len(report.submitted) == len(groups)
    assert len(read_jobs(path=folder / 'submissions.jsonl')) == len(groups)
    logs = ''.join(path.read_text() for path in (folder / 'logs').glob('*.out'))
    assert logs.count('could not start') == sum(len(group.tasks) for group in groups)


def _local_groups(study: Study, machine: MachineProfile, *, local_cpus: int = 1) -> list[JobGroup]:
    runs = expand_runs(study=study)
    resolved = resolve_runs(study=study, machine=machine, runs=runs, local_cpus=local_cpus)
    return plan_jobs(
        study=study,
        machine=machine,
        resolved=resolved,
        indices=assign_indices(runs=runs, jobs=[]),
        extra_args={},
        attempts={},
        first_job_file=1,
        environment={},
    )


def _times_script(tmp_path: Path) -> Path:
    script = tmp_path / 'times.py'
    script.write_text(
        'import hashlib\n'
        'import os\n'
        'import sys\n'
        'import time\n'
        'args = sys.argv[1:]\n'
        'run_path = args[args.index("--run-path") + 1]\n'
        'key = hashlib.sha1(run_path.encode()).hexdigest()[:12]\n'
        'cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "none")\n'
        'path = os.path.join(os.environ["TIMES_DIR"], key + ".times")\n'
        'with open(path, "w") as f:\n'
        '    f.write(f"{time.time()}\\n{cuda}\\n")\n'
        'time.sleep(1)\n'
        'with open(path, "a") as f:\n'
        '    f.write(f"{time.time()}\\n")\n'
    )
    return script


def _local_machine(machine: MachineProfile) -> MachineProfile:
    return MachineProfile(
        path=machine.path, launcher='local', output_root=machine.output_root, slurm=None
    )


def test_local_max_parallel_three_overlaps_at_most_three(
    machine: MachineProfile, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_machine = _local_machine(machine)
    times = tmp_path / 'times'
    times.mkdir()
    monkeypatch.setenv('TIMES_DIR', str(times))
    study = Study(
        name='s',
        grids={
            'g': grid(
                short_name='g1',
                command=(sys.executable, str(_times_script(tmp_path))),
                axes={'seed': setting_axis(path='seed', values=(0, 1, 2, 3, 4))},
            )
        },
    )
    local = LocalSettings(max_parallel=3)
    report = launch(
        launcher=LocalLauncher(project_root=tmp_path, env={}, local=local),
        machine=local_machine,
        study=study,
        groups=_local_groups(study, local_machine),
        project_root=tmp_path,
        dry_run=False,
    )
    assert len(report.submitted) == 1
    folder = study_folder(machine=local_machine, study=study)
    files = sorted(times.glob('*.times'))
    assert len(files) == 5
    events: list[tuple[float, int]] = []
    for path in files:
        start, _cuda, end = path.read_text().splitlines()
        events += [(float(start), 1), (float(end), -1)]
    events.sort(key=lambda event: (event[0], event[1]))
    peak = current = 0
    for _, delta in events:
        current += delta
        peak = max(peak, current)
    assert peak == 3
    usages = [json.loads(path.read_text()) for path in (folder / 'logs').glob('*.usage.json')]
    assert len(usages) == 5
    assert all(usage['exit_code'] == 0 for usage in usages)


def test_local_gpu_ids_reach_tasks(
    machine: MachineProfile, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_machine = _local_machine(machine)
    times = tmp_path / 'times'
    times.mkdir()
    monkeypatch.setenv('TIMES_DIR', str(times))
    monkeypatch.setenv('XDG_RUNTIME_DIR', str(tmp_path / 'xdg'))
    study = Study(
        name='s',
        grids={'g': grid(short_name='g1', command=(sys.executable, str(_times_script(tmp_path))))},
    )
    local = LocalSettings(gpus=(0, 1), tasks_per_gpu=2)
    launch(
        launcher=LocalLauncher(project_root=tmp_path, env={}, local=local, gpu_ids=('0', '1')),
        machine=local_machine,
        study=study,
        groups=_local_groups(study, local_machine),
        project_root=tmp_path,
        dry_run=False,
    )
    folder = study_folder(machine=local_machine, study=study)
    files = sorted(times.glob('*.times'))
    assert len(files) == 4
    for path in files:
        _start, cuda, _end = path.read_text().splitlines()
        assert cuda in ('0', '1')
    headers = [path.read_text().splitlines()[0] for path in (folder / 'logs').glob('*.out')]
    assert len(headers) == 4
    assert all('| gpus 0,1' in header for header in headers)


def test_launcher_logs_a_busy_gpu_once(
    machine: MachineProfile,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    local_machine = _local_machine(machine)
    script = tmp_path / 'busy.py'
    script.write_text('import time\ntime.sleep(60)\n')
    study = Study(
        name='s',
        grids={
            'g': grid(
                short_name='g1',
                command=(sys.executable, str(script)),
                axes={'seed': setting_axis(path='seed', values=(0,))},
            )
        },
    )

    def busy_gpus(*, gpu_ids: Sequence[str], memory_fraction: float = 1.0) -> frozenset[str]:
        return frozenset(gpu_ids)

    monkeypatch.setattr(_local_run, 'busy_gpus', busy_gpus)
    monkeypatch.setenv('XDG_RUNTIME_DIR', str(tmp_path / 'xdg'))
    local = LocalSettings(gpus=(0,), skip_busy_gpus=True)
    timer = threading.Timer(0.5, lambda: os.kill(os.getpid(), signal.SIGTERM))
    timer.start()
    with caplog.at_level(logging.INFO):
        report = launch(
            launcher=LocalLauncher(project_root=tmp_path, env={}, local=local, gpu_ids=('0',)),
            machine=local_machine,
            study=study,
            groups=_local_groups(study, local_machine),
            project_root=tmp_path,
            dry_run=False,
        )
    timer.cancel()
    assert report.stopped_signal == signal.SIGTERM
    busy_lines = [
        record.getMessage() for record in caplog.records if 'GPU 0 is busy' in record.getMessage()
    ]
    assert busy_lines == ['GPU 0 is busy; waiting for it']


def test_busy_gpu_log_is_not_repeated_after_the_recheck_window(
    machine: MachineProfile,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    local_machine = _local_machine(machine)
    script = tmp_path / 'busy.py'
    script.write_text('import time\ntime.sleep(60)\n')
    study = Study(
        name='s',
        grids={
            'g': grid(
                short_name='g1',
                command=(sys.executable, str(script)),
                axes={'seed': setting_axis(path='seed', values=(0,))},
            )
        },
    )

    def busy_gpus(*, gpu_ids: Sequence[str], memory_fraction: float = 1.0) -> frozenset[str]:
        return frozenset(gpu_ids)

    # why: a short recheck window forces the pool to recheck the same busy GPU
    monkeypatch.setattr(
        _local_run,
        'run_commands',
        functools.partial(_local_run.run_commands, busy_recheck_seconds=0.2, gpu_poll_seconds=0.1),
    )
    monkeypatch.setattr(_local_run, 'busy_gpus', busy_gpus)
    monkeypatch.setenv('XDG_RUNTIME_DIR', str(tmp_path / 'xdg'))
    local = LocalSettings(gpus=(0,), skip_busy_gpus=True)
    timer = threading.Timer(1.5, lambda: os.kill(os.getpid(), signal.SIGTERM))
    timer.start()
    with caplog.at_level(logging.INFO):
        report = launch(
            launcher=LocalLauncher(project_root=tmp_path, env={}, local=local, gpu_ids=('0',)),
            machine=local_machine,
            study=study,
            groups=_local_groups(study, local_machine),
            project_root=tmp_path,
            dry_run=False,
        )
    timer.cancel()
    assert report.stopped_signal == signal.SIGTERM
    busy_lines = [
        record.getMessage() for record in caplog.records if 'GPU 0 is busy' in record.getMessage()
    ]
    assert busy_lines == ['GPU 0 is busy; waiting for it']


def _omp_machine(machine: MachineProfile) -> MachineProfile:
    return MachineProfile(
        path=machine.path,
        launcher='local',
        output_root=machine.output_root,
        local=LocalSettings(max_parallel=2),
    )


def _omp_study(tmp_path: Path) -> Study:
    script = tmp_path / 'omp.py'
    script.write_text("import os; print(os.environ['OMP_NUM_THREADS'])\n")
    return Study(
        name='s', grids={'g': grid(short_name='g1', command=(sys.executable, str(script)))}
    )


def test_omp_num_threads_is_set_for_local_runs(
    machine: MachineProfile, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv('OMP_NUM_THREADS', raising=False)
    local_machine = _omp_machine(machine)
    study = _omp_study(tmp_path)
    launch(
        launcher=LocalLauncher(project_root=tmp_path, env={}, local=LocalSettings(max_parallel=2)),
        machine=local_machine,
        study=study,
        groups=_local_groups(study, local_machine, local_cpus=4),
        project_root=tmp_path,
        dry_run=False,
    )
    folder = study_folder(machine=local_machine, study=study)
    assert '2' in (folder / 'logs' / 'g1_0.out').read_text().splitlines()


def test_preset_omp_num_threads_is_kept(
    machine: MachineProfile, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv('OMP_NUM_THREADS', raising=False)
    local_machine = _omp_machine(machine)
    study = _omp_study(tmp_path)
    launch(
        launcher=LocalLauncher(
            project_root=tmp_path, env={'OMP_NUM_THREADS': '1'}, local=LocalSettings(max_parallel=2)
        ),
        machine=local_machine,
        study=study,
        groups=_local_groups(study, local_machine, local_cpus=4),
        project_root=tmp_path,
        dry_run=False,
    )
    folder = study_folder(machine=local_machine, study=study)
    assert '1' in (folder / 'logs' / 'g1_0.out').read_text().splitlines()


def test_local_two_launchers_never_share_a_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('XDG_RUNTIME_DIR', str(tmp_path / 'xdg'))
    worker = (
        'import sys\n'
        'from pathlib import Path\n'
        'from tslab.study.definition import StudyRun\n'
        'from tslab.study.launchers.local.run import LocalLauncher\n'
        'from tslab.study.launchers.local.backend import LocalResources\n'
        'from tslab.study.machine import LocalSettings\n'
        'from tslab.study.plan import JobGroup, PlannedTask, ResolvedRun\n'
        'root = Path(sys.argv[1])\n'
        'index = sys.argv[2]\n'
        'inner = (\n'
        '    "import os, sys, time; base = os.path.join(sys.argv[1], sys.argv[2]); "\n'
        "    \"open(base + '.s', 'w').write(str(time.time())); time.sleep(2); \"\n"
        "    \"open(base + '.e', 'w').write(str(time.time()))\"\n"
        ')\n'
        'task = PlannedTask(\n'
        '    resolved=ResolvedRun(\n'
        '        run=StudyRun(grid="g", values={}, run_path="p", settings={}),\n'
        '        settings={},\n'
        '        resources=LocalResources(cpus=1, cpus_per_run=1),\n'
        '    ),\n'
        '    index=1,\n'
        '    log_file=root / ("task" + index + ".log"),\n'
        '    command=(sys.executable, "-c", inner, str(root), index),\n'
        '    extra_args=(),\n'
        '    attempt=1,\n'
        ')\n'
        'group = JobGroup(\n'
        '    grid="g", job_name="g", resources=LocalResources(cpus=1, cpus_per_run=1),\n'
        '    job_file=1, tasks=(task,)\n'
        ')\n'
        'LocalLauncher(project_root=root, env={}, local=LocalSettings(), gpu_ids=("0",), '
        'pid=123, host="test-host", now=lambda: "2026-01-02T03:04:05+00:00").submit(\n'
        '    group=group, study_folder=root, record=lambda job_id: None\n'
        ')\n'
    )
    procs = [
        subprocess.Popen(  # noqa: S603 # why: the command is fixed by the test
            [sys.executable, '-c', worker, str(tmp_path), str(index)]
        )
        for index in ('1', '2')
    ]
    for proc in procs:
        assert proc.wait(timeout=60) == 0
    intervals = sorted(
        (float((tmp_path / f'{index}.s').read_text()), float((tmp_path / f'{index}.e').read_text()))
        for index in ('1', '2')
    )
    assert intervals[0][1] <= intervals[1][0]


def test_local_timeout_sends_usr2_and_logs_steps(machine: MachineProfile, tmp_path: Path) -> None:
    local_machine = _local_machine(machine)
    script = tmp_path / 'usr2.py'
    script.write_text(
        'import signal\n'
        'import sys\n'
        'import time\n'
        'def done(signum, frame):\n'
        '    sys.exit(0)\n'
        'signal.signal(signal.SIGUSR2, done)\n'
        'while True:\n'
        '    time.sleep(0.05)\n'
    )
    study = Study(
        name='s',
        grids={
            'g': grid(
                short_name='g1',
                command=(sys.executable, str(script)),
                axes={'seed': setting_axis(path='seed', values=(0,))},
            )
        },
    )
    local = LocalSettings(timeout='0:03', signal_seconds=1)
    launch(
        launcher=LocalLauncher(project_root=tmp_path, env={}, local=local),
        machine=local_machine,
        study=study,
        groups=_local_groups(study, local_machine),
        project_root=tmp_path,
        dry_run=False,
    )
    folder = study_folder(machine=local_machine, study=study)
    log = (folder / 'logs' / 'g1_0.out').read_text()
    assert 'sent USR2' in log
    usage = json.loads((folder / 'logs' / 'g1_0.out.usage.json').read_text())
    assert usage['exit_code'] == 0
    assert usage['timed_out'] is False
    assert usage['timeout'] == 3.0


def test_local_parallel_console_shows_one_line_per_start_and_end(
    machine: MachineProfile, tmp_path: Path
) -> None:
    local_machine = _local_machine(machine)
    script = tmp_path / 'echo.py'
    script.write_text("print('TASK-OUTPUT')\nimport time\ntime.sleep(0.5)\n")
    study = Study(
        name='s',
        grids={
            'g': grid(
                short_name='g1',
                command=(sys.executable, str(script)),
                axes={'seed': setting_axis(path='seed', values=(0, 1))},
            )
        },
    )
    stream = StringIO()
    local = LocalSettings(max_parallel=2)
    launch(
        launcher=LocalLauncher(project_root=tmp_path, env={}, local=local, stream=stream),
        machine=local_machine,
        study=study,
        groups=_local_groups(study, local_machine),
        project_root=tmp_path,
        dry_run=False,
    )
    console = stream.getvalue()
    assert 'TASK-OUTPUT' not in console
    assert console.count(' started |') == 2
    assert console.count(' ended |') == 2
    assert 'gpu none' in console
    assert 'exit 0' in console


def test_local_usage_file_and_footer(machine: MachineProfile, tmp_path: Path) -> None:
    local_machine = _local_machine(machine)
    script = tmp_path / 'alloc.py'
    script.write_text(
        'import sys\n'
        'import time\n'
        'args = sys.argv[1:]\n'
        "run_path = args[args.index('--run-path') + 1]\n"
        'held = bytearray(100 * 1024 * 1024) if run_path.endswith("/100") else bytearray(1024)\n'
        'time.sleep(1)\n'
        'del held\n'
    )
    study = Study(
        name='s',
        grids={
            'g': grid(
                short_name='g1',
                command=(sys.executable, str(script)),
                axes={'seed': setting_axis(path='seed', values=(100, 1))},
            )
        },
    )
    local = LocalSettings()
    launch(
        launcher=LocalLauncher(project_root=tmp_path, env={}, local=local),
        machine=local_machine,
        study=study,
        groups=_local_groups(study, local_machine),
        project_root=tmp_path,
        dry_run=False,
    )
    folder = study_folder(machine=local_machine, study=study)
    usages = sorted(
        (json.loads(path.read_text()) for path in (folder / 'logs').glob('*.usage.json')),
        key=lambda usage: usage['max_rss_bytes'],
    )
    assert len(usages) == 2
    assert all(usage['elapsed_seconds'] >= 1 for usage in usages)
    assert usages[1]['max_rss_bytes'] >= 100 * 1024 * 1024
    assert usages[1]['max_rss_bytes'] > usages[0]['max_rss_bytes']
    for log in (folder / 'logs').glob('*.out'):
        assert log.read_text().strip().splitlines()[-1].startswith('=== task')


_SIGNAL_SCRIPT = (
    'import signal\n'
    'import sys\n'
    'import time\n'
    'signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))\n'
    'while True:\n'
    '    time.sleep(0.1)\n'
)

_IGNORE_SIGTERM_SCRIPT = (
    'import signal\n'
    'import time\n'
    'signal.signal(signal.SIGTERM, signal.SIG_IGN)\n'
    'while True:\n'
    '    time.sleep(0.1)\n'
)


def test_local_sigterm_stops_tasks_and_reports_signal(
    machine: MachineProfile, tmp_path: Path
) -> None:
    local_machine = _local_machine(machine)
    study = Study(
        name='s',
        grids={
            'a': grid(
                short_name='g1',
                command=(sys.executable, '-c', _SIGNAL_SCRIPT),
                axes={'seed': setting_axis(path='seed', values=(0,))},
            ),
            'b': grid(
                short_name='g2',
                command=(sys.executable, '-c', _SIGNAL_SCRIPT),
                axes={'seed': setting_axis(path='seed', values=(0,))},
            ),
        },
    )
    timer = threading.Timer(0.5, lambda: os.kill(os.getpid(), signal.SIGTERM))
    timer.start()
    start = time.monotonic()
    try:
        report = launch(
            launcher=LocalLauncher(project_root=tmp_path, env={}, local=LocalSettings()),
            machine=local_machine,
            study=study,
            groups=_local_groups(study, local_machine),
            project_root=tmp_path,
            dry_run=False,
        )
    finally:
        timer.cancel()
    elapsed = time.monotonic() - start
    assert report.stopped_signal == signal.SIGTERM
    assert [group.job_name for group in report.not_submitted] == ['g2']
    assert len(report.submitted) == 1
    assert elapsed < 5


def test_local_stop_kills_task_ignoring_sigterm(machine: MachineProfile, tmp_path: Path) -> None:
    local_machine = _local_machine(machine)
    study = Study(
        name='s',
        grids={
            'g': grid(
                short_name='g1',
                command=(sys.executable, '-c', _IGNORE_SIGTERM_SCRIPT),
                axes={'seed': setting_axis(path='seed', values=(0,))},
            )
        },
    )
    timer = threading.Timer(0.5, lambda: os.kill(os.getpid(), signal.SIGTERM))
    timer.start()
    start = time.monotonic()
    try:
        report = launch(
            launcher=LocalLauncher(
                project_root=tmp_path, env={}, local=LocalSettings(), grace_seconds=1
            ),
            machine=local_machine,
            study=study,
            groups=_local_groups(study, local_machine),
            project_root=tmp_path,
            dry_run=False,
        )
    finally:
        timer.cancel()
    elapsed = time.monotonic() - start
    assert report.stopped_signal == signal.SIGTERM
    assert elapsed < 4
    folder = study_folder(machine=local_machine, study=study)
    assert 'exit -9' in (folder / 'logs' / 'g1_0.out').read_text()


class _FakeWatchdog:
    """A stand-in for the launcher's watchdog that records every call."""

    started: ClassVar[list['_FakeWatchdog']] = []

    def __init__(self, *, process: subprocess.Popen[bytes], grace_seconds: float) -> None:
        self.process = process
        self.grace_seconds = grace_seconds
        self.watched: list[int] = []
        self.forgotten: list[int] = []
        self.closed = False

    @classmethod
    def start(cls, *, grace_seconds: float) -> '_FakeWatchdog':
        process = subprocess.Popen([sys.executable, '-c', 'pass'])  # why: a fixed test command
        process.wait()
        fake = cls(process=process, grace_seconds=grace_seconds)
        cls.started.append(fake)
        return fake

    def watch(self, *, pid: int) -> None:
        self.watched.append(pid)

    def forget(self, *, pid: int) -> None:
        self.forgotten.append(pid)

    def close(self) -> None:
        self.closed = True
        self.process.wait()


def test_normal_launch_end_leaves_the_runs_alone(
    machine: MachineProfile, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_machine = _local_machine(machine)
    _FakeWatchdog.started.clear()
    monkeypatch.setattr(_local_run, 'Watchdog', _FakeWatchdog)
    study = Study(
        name='s',
        grids={
            'g': grid(
                short_name='g1',
                command=(sys.executable, '-c', 'pass'),
                axes={'seed': setting_axis(path='seed', values=(0, 1))},
            )
        },
    )
    launch(
        launcher=LocalLauncher(project_root=tmp_path, env={}, local=LocalSettings(max_parallel=2)),
        machine=local_machine,
        study=study,
        groups=_local_groups(study, local_machine),
        project_root=tmp_path,
        dry_run=False,
    )
    assert len(_FakeWatchdog.started) == 1
    fake = _FakeWatchdog.started[0]
    assert fake.closed is True
    assert fake.process.poll() is not None
    assert len(fake.watched) == 2
    assert sorted(fake.watched) == sorted(fake.forgotten)


def _wait_for_run_pid(*, launcher: subprocess.Popen[bytes], launch_file: Path) -> int:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if launcher.poll() is not None:
            message = f'the launcher exited with {launcher.returncode} before a run started'
            raise AssertionError(message)
        document = json.loads(launch_file.read_text()) if launch_file.exists() else {}
        pids = document.get('run_pids') or {}
        if pids:
            return int(next(iter(pids.values())))
        time.sleep(0.2)
    message = 'no run pid in the launch file after 30 seconds'
    raise AssertionError(message)


def _group_exists(*, pid: int) -> bool:
    try:
        os.killpg(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


@pytest.mark.slow
def test_launcher_crash_stops_its_runs(project: Path) -> None:
    launcher = subprocess.Popen(  # why: the study submit of the toy project
        [
            sys.executable,
            '-m',
            'tslab',
            'study',
            'submit',
            'toy_studies:study',
            '--machine',
            'local.toml',
        ],
        cwd=project,
        env={**os.environ, 'TOY_SLEEP': '600'},
        start_new_session=True,
    )
    launch_file = study_directory(project=project) / '.study' / LAUNCH_FILE
    try:
        run_pid = _wait_for_run_pid(launcher=launcher, launch_file=launch_file)
        os.kill(launcher.pid, signal.SIGKILL)
        deadline = time.monotonic() + 10
        while _group_exists(pid=run_pid) and time.monotonic() < deadline:
            time.sleep(0.2)
        assert not _group_exists(pid=run_pid), (
            f'the run {run_pid} survived the crash of its launcher'
        )
        assert launcher.wait(timeout=30) is not None
    finally:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(launcher.pid, signal.SIGKILL)
        launcher.wait()

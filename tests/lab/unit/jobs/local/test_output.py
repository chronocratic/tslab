import io
import json
from pathlib import Path

from tslab.lab.jobs.local import ProcessResult, tail_lines, TaskInfo, TaskOutput


def test_tail_lines_returns_last_lines(*, tmp_path: Path) -> None:
    path = tmp_path / 'launch.log'
    path.write_text('first\nsecond\nthird\nfourth\n', encoding='utf-8')
    assert tail_lines(path=path, lines=2) == ['third', 'fourth']
    assert tail_lines(path=path, lines=10) == ['first', 'second', 'third', 'fourth']


def test_tail_lines_of_missing_file_is_none(*, tmp_path: Path) -> None:
    assert tail_lines(path=tmp_path / 'missing.log', lines=5) is None


def test_task_output_writes_header_and_lines(*, tmp_path: Path) -> None:
    stream = io.StringIO()
    log_file = tmp_path / 'logs' / '0.out'
    usage_file = tmp_path / 'logs' / '0.out.usage.json'
    states: list[tuple[str, str]] = []
    output = TaskOutput(
        stream=stream,
        single_slot=True,
        tasks={
            'k_0': TaskInfo(
                log_file=log_file,
                index=0,
                name='run s/cost',
                usage_file=usage_file,
                usage_document={'schema': 'usage/1', 'attempt': 1},
            )
        },
        header=lambda key: f'=== attempt 1 | job j | {key} ===\n',
        on_task_state=lambda key, state: states.append((key, state)),
    )
    output.on_start(key='k_0', gpu_id=None)
    output.on_line(key='k_0', line='hello\n')
    output.on_end(
        key='k_0',
        gpu_id=None,
        outcome=ProcessResult(
            exit_code=0, elapsed_seconds=1.5, cpu_seconds=1.0, max_rss_bytes=1024, timed_out=False
        ),
    )
    text = log_file.read_text(encoding='utf-8')
    assert text.startswith('=== attempt 1 | job j | k_0 ===\n')
    assert 'hello\n' in text
    assert text.rstrip().splitlines()[-1].startswith('=== task 0: exit 0')
    usage = json.loads(usage_file.read_text(encoding='utf-8'))
    assert usage == {
        'schema': 'usage/1',
        'attempt': 1,
        'exit_code': 0,
        'elapsed_seconds': 1.5,
        'cpu_seconds': 1.0,
        'max_rss_bytes': 1024,
        'gpu_id': None,
        'timed_out': False,
    }
    assert states == [('k_0', 'running'), ('k_0', 'ended')]
    console = stream.getvalue()
    assert console.startswith('hello\n')
    assert console.endswith('=== task 0: exit 0 | elapsed 0:00:01 | peak memory 1.0 KiB ===\n')

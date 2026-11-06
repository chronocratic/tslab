"""SLURM job scripts for study jobs."""

from collections.abc import Mapping, Sequence
from pathlib import Path
import shlex

from tslab.jobs.slurm import (
    array_spec,
    check_script_text,
    export_lines,
    node_local_tmp_lines,
    sbatch_directives,
    srun_command,
    wait_for_file_line,
)
from tslab.study.code_snapshot import CodeSnapshot, SRC_FOLDER
from tslab.study.machine import SlurmSettings

__all__ = ['foreground_script', 'job_script']


def foreground_script(
    *, slurm: SlurmSettings, project_root: Path, env: Mapping[str, str], command: Sequence[str]
) -> str:
    """The shell script that runs one study run in the foreground.

    The same pieces and order as ``job_script``, without the ``#SBATCH``
    lines, the srun task section, and the node-local temporary folder: the
    command runs as the one process of the terminal.

    Args:
        slurm: The resource settings of the machine, for its ``setup`` lines.
        project_root: The directory the command runs in.
        env: Variables to export before the command.
        command: The command to run; it replaces the script via ``exec``.

    Returns:
        The script text.

    Raises:
        ValueError: If a ``setup`` line holds a line break or other control
            character, or a name in ``env`` is not a shell variable name.
    """
    for line in slurm.setup:
        check_script_text(value=line, what='setup')
    lines = [
        '#!/bin/bash',
        'set -euo pipefail',
        f'cd {shlex.quote(str(project_root))}',
        # why: a killed run would lose the output still held in the buffer
        'export PYTHONUNBUFFERED=1',
    ]
    lines += export_lines(env=env)
    lines += list(slurm.setup)
    lines.append(f'exec {shlex.join(command)}')
    return '\n'.join(lines) + '\n'


def _wait_section(*, tasks_file: Path) -> str:
    # why: on shared storage the file written by the launcher can reach the node late
    return f'{wait_for_file_line(path=tasks_file, seconds=60)}\n'


def _array_section(
    *, tasks_file: Path, chain_check: Sequence[str] | None, requeue: bool | str
) -> str:
    # why: a requeued task starts again with the same line, so the restart count adds to the attempt
    attempt = '$((ATTEMPT + ${SLURM_RESTART_COUNT:-0}))' if requeue else '${ATTEMPT}'
    section = _wait_section(tasks_file=tasks_file) + (
        'TASK_LINE=$(sed -n "s/^${SLURM_ARRAY_TASK_ID} //p" ' + shlex.quote(str(tasks_file)) + ')\n'
        'if [ -z "$TASK_LINE" ]; then\n'
        '    echo "no task ${SLURM_ARRAY_TASK_ID} in "' + shlex.quote(str(tasks_file)) + ' >&2\n'
        '    exit 1\n'
        'fi\n'
        'ATTEMPT=${TASK_LINE%% *}\n'
        'TASK_LINE=${TASK_LINE#* }\n'
        'echo "=== tslab attempt ' + attempt + ' | job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}'
        ' | $(hostname) | $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="\n'
    )
    if chain_check is None:
        return section + srun_command(variable='TASK_LINE')
    command = shlex.join(chain_check)
    return (
        section
        + 'set +e\n'
        + srun_command(variable='TASK_LINE')
        + '\nRUN_EXIT=$?\n'
        + f'{command} --index ${{SLURM_ARRAY_TASK_ID}}\n'
        + 'CHECK_EXIT=$?\n'
        + 'if [ "$RUN_EXIT" -ne 0 ]; then\n'
        + '    exit "$RUN_EXIT"\n'
        + 'fi\n'
        + 'exit "$CHECK_EXIT"\n'
    )


def _single_section(*, tasks_file: Path) -> str:
    return _wait_section(tasks_file=tasks_file) + (
        'TASK_LINE=$(sed -n 1p ' + shlex.quote(str(tasks_file)) + ')\n'
        # why: like the array section, the line starts with the index and the attempt
        'TASK_LINE=${TASK_LINE#* }\n'
        'TASK_LINE=${TASK_LINE#* }\n'
        'echo "=== tslab job ${SLURM_JOB_ID} | $(hostname) | $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="\n'
        + srun_command(variable='TASK_LINE')
    )


def _check_job_values(  # why: keyword-only private helper, one per checked value
    *,
    job_name: str,
    slurm: SlurmSettings,
    log_pattern: Path,
    dependency: str | None,
    begin: str | None,
) -> None:
    for what, value in (
        ('job_name', job_name),
        ('partition', slurm.partition),
        ('qos', slurm.qos or ''),
        ('time', slurm.time),
        ('mem_per_cpu', slurm.mem_per_cpu or ''),
        ('gpu_type', slurm.gpu_type or ''),
        ('log_pattern', log_pattern),
        ('dependency', dependency or ''),
        ('begin', begin or ''),
        *((f'sbatch_extra.{key}', value) for key, value in slurm.sbatch_extra),
        *(('setup', line) for line in slurm.setup),
    ):
        check_script_text(value=value, what=what)


def _gres(*, slurm: SlurmSettings) -> str:
    """The ``--gres`` value of the GPUs of one task."""
    if slurm.gpu_type is None:
        return f'gpu:{slurm.gpus}'
    return f'gpu:{slurm.gpu_type}:{slurm.gpus}'


def job_script(  # noqa: PLR0913 # why: keyword-only public API; one option per script line
    *,
    job_name: str,
    slurm: SlurmSettings,
    indices: Sequence[int] | None,
    tasks_file: Path,
    log_pattern: Path,
    project_root: Path,
    env: Mapping[str, str],
    dependency: str | None = None,
    begin: str | None = None,
    chain_check: Sequence[str] | None = None,
    requeue: bool | str = False,
    code_snapshot: CodeSnapshot | None = None,
) -> str:
    """The full ``.sbatch`` script of one study job.

    Args:
        job_name: The ``--job-name``; kept short so logs stay readable.
        slurm: The resource settings of the job.
        indices: The array indices, or ``None`` for a single (follow-up) job.
        tasks_file: The ``.tasks`` file the script reads its command from.
        log_pattern: The ``--output`` pattern; SLURM fills in ``%a``.
        project_root: The directory the command runs in.
        env: Variables to export before the command.
        dependency: A ``--dependency`` value, when given.
        begin: A ``--begin`` value, when given.
        chain_check: The run-state check after ``srun``; the array index is added.
        requeue: ``False`` writes ``--no-requeue``; ``True`` writes ``--requeue``; the
            string ``'expedite'`` writes ``--requeue=expedite``.
        code_snapshot: The code snapshot the job runs from; the script checks
            its ``src`` folder, exports its variables and puts it first on
            ``PYTHONPATH``. ``None`` runs the live code.

    Returns:
        The script text, ``#SBATCH`` lines first.

    Raises:
        ValueError: If a value written to an ``#SBATCH`` line holds a line
            break or other control character, or a name in ``env`` is not a
            shell variable name.
    """
    _check_job_values(
        job_name=job_name, slurm=slurm, log_pattern=log_pattern, dependency=dependency, begin=begin
    )
    array = indices is not None
    options = _job_options(
        job_name=job_name,
        slurm=slurm,
        indices=indices,
        log_pattern=log_pattern,
        dependency=dependency,
        begin=begin,
        chain_check=chain_check,
        requeue=requeue,
    )
    lines = ['#!/bin/bash', *sbatch_directives(options=options)]
    lines += [
        'set -euo pipefail',
        f'cd {shlex.quote(str(project_root))}',
        # why: a killed job would lose the output still held in the buffer
        'export PYTHONUNBUFFERED=1',
    ]
    lines += export_lines(env=env)
    lines += list(slurm.setup)
    if slurm.node_local_tmp:
        lines += node_local_tmp_lines(prefix='tslab', array=array)
        lines.extend(
            [
                'export WANDB_CACHE_DIR=$TMPDIR/wandb-cache',
                'export WANDB_CONFIG_DIR=$TMPDIR/wandb-config',
            ]
        )
    if code_snapshot is not None:
        lines += _code_snapshot_lines(code_snapshot=code_snapshot)
    if chain_check is not None and indices is None:
        message = 'chain_check requires an array job'
        raise ValueError(message)
    if indices is None:
        lines.append(_single_section(tasks_file=tasks_file))
    else:
        lines.append(
            _array_section(tasks_file=tasks_file, chain_check=chain_check, requeue=requeue)
        )
    return '\n'.join(lines) + '\n'


def _job_options(  # noqa: PLR0913 # why: options depend on seven independent script fields
    *,
    job_name: str,
    slurm: SlurmSettings,
    indices: Sequence[int] | None,
    log_pattern: Path,
    dependency: str | None,
    begin: str | None,
    chain_check: Sequence[str] | None,
    requeue: bool | str,
) -> dict[str, object]:
    """The sbatch options shared by array groups and helper jobs."""
    options: dict[str, object] = {'job_name': job_name, 'partition': slurm.partition}
    if slurm.qos is not None:
        options['qos'] = slurm.qos
    options.update(
        time=slurm.time,
        nodes=1,
        ntasks_per_node=max(slurm.gpus, 1),
        cpus_per_task=slurm.cpus_per_task,
    )
    if slurm.mem_per_cpu is not None:
        options['mem_per_cpu'] = slurm.mem_per_cpu
    if slurm.gpus > 0:
        options['gres'] = _gres(slurm=slurm)
    if indices is not None:
        spec = f'{array_spec(indices=indices)}%{slurm.max_concurrent}'
        options['array'] = spec
    options.update(output=log_pattern, open_mode='append')
    if requeue:
        options['requeue'] = requeue
    else:
        options['no_requeue'] = True
    if slurm.signal_seconds > 0:
        options['signal'] = f'USR2@{slurm.signal_seconds}'
    if dependency is not None:
        options['dependency'] = dependency
    if begin is not None:
        options['begin'] = begin
    options.update((key.replace('-', '_'), value) for key, value in slurm.sbatch_extra)
    if chain_check is not None:
        options['kill_on_invalid_dep'] = 'yes'
    return options


def _code_snapshot_lines(*, code_snapshot: CodeSnapshot) -> list[str]:
    # why: the job must not silently fall back to the live code if the snapshot was removed
    source = str(code_snapshot.folder / SRC_FOLDER)
    return [
        f'if [ ! -d {shlex.quote(source)} ]; then',
        f'    echo "tslab: code snapshot {source} is missing; not running the live code" >&2',
        '    exit 1',
        'fi',
        *export_lines(env=code_snapshot.variables()),
        f'export PYTHONPATH={shlex.quote(source)}${{PYTHONPATH:+:$PYTHONPATH}}',
    ]

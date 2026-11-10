import subprocess
import sys
import time

import pytest

from tslab.lab.jobs.local.watchdog import _apply_line, main, Watchdog


def _group_leader() -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        ['sleep', '600'],  # noqa: S607 # why: a fixed system command for the test
        start_new_session=True,
    )


def test_closing_with_a_watched_run_kills_its_group() -> None:
    run = _group_leader()
    watchdog = Watchdog.start(grace_seconds=1.0)
    watchdog.watch(pid=run.pid)
    watchdog.close()
    assert run.wait(timeout=10) is not None


def test_close_after_forget_kills_nothing() -> None:
    run = _group_leader()
    watchdog = Watchdog.start(grace_seconds=1.0)
    watchdog.watch(pid=run.pid)
    watchdog.forget(pid=run.pid)
    watchdog.close()
    assert run.poll() is None
    run.kill()
    run.wait()


def test_apply_line_does_not_mutate_its_set() -> None:
    watched = {1}
    added = _apply_line(line='+2\n', watched=watched)
    dropped = _apply_line(line='-1\n', watched=watched)
    assert watched == {1}
    assert added == {1, 2}
    assert dropped == set()


def test_send_requires_keyword_argument() -> None:
    process = subprocess.Popen([sys.executable, '-c', 'pass'])
    process.wait()
    watchdog = Watchdog(process=process, grace_seconds=1.0)
    with pytest.raises(TypeError):
        # ty: ignore[missing-argument, too-many-positional-arguments]
        watchdog._send('+1\n')  # noqa: SLF001 # why: the test checks the keyword-only contract


def test_a_group_already_gone_is_ignored() -> None:
    assert main(argv=['0.1'], lines=['+999999\n']) == 0


def test_a_run_that_ignores_term_gets_killed() -> None:
    command = ['bash', '-c', 'trap "" TERM; sleep 600']  # why: a fixed test command
    run = subprocess.Popen(command, start_new_session=True)  # noqa: S603 # why: a fixed test command
    # why: the trap must be set before the watchdog sends the TERM
    time.sleep(0.2)
    assert main(argv=['0.5'], lines=[f'+{run.pid}\n']) == 0
    assert run.wait(timeout=10) == -9

"""SLURM command and cluster readiness checks."""

import os
from pathlib import Path

import pytest

from tests.support.fake_slurm import FakeSlurm
from tslab.study.doctor import DoctorCheck, run_checks
from tslab.study.launchers.slurm import doctor
from tslab.study.machine import SlurmSettings


def test_slurm_check_lists_missing_scheduler_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    def which(name: str) -> str | None:
        return None if name == 'sacct' else '/bin/' + name

    monkeypatch.setattr(doctor.shutil, 'which', which)

    check = doctor.slurm_check()

    assert (check.name, check.status) == ('slurm', 'fail')
    assert check.message == 'missing SLURM commands: sacct'


def test_doctor_fails_missing_slurm_commands(
    project: Path, fake: FakeSlurm, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    empty_path = tmp_path / 'empty-bin'
    empty_path.mkdir()
    monkeypatch.setenv('PATH', str(empty_path))
    checks = run_checks(
        machine_path=project / 'cluster.toml', study=None, start=project, environment=os.environ
    )
    slurm = next(check for check in checks if check.name == 'slurm')
    assert slurm.status == 'fail'
    assert 'sbatch' in slurm.message
    assert 'squeue' in slurm.message
    assert 'sacct' in slurm.message


def _check(*, checks: tuple[DoctorCheck, ...], name: str) -> DoctorCheck:
    return next(check for check in checks if check.name == name)


def test_array_size_above_the_cluster_limit_fails(fake: FakeSlurm) -> None:
    fake.set_answers('scontrol', [{'exit': 0, 'stdout': 'MaxArraySize            = 1001\n'}])
    fake.set_answers('sinfo', [{'exit': 0, 'stdout': 'infinite|32|128000|(null)\n'}])
    checks = doctor.cluster_checks(slurm=SlurmSettings(partition='gpu', max_array_size=2000))
    check = _check(checks=checks, name='slurm-array')
    assert check.status == 'fail'
    assert (
        check.message
        == 'max_array_size 2000 is above the cluster limit: MaxArraySize 1001 allows 1000'
    )


def test_array_size_at_the_cluster_limit_passes(fake: FakeSlurm) -> None:
    fake.set_answers('scontrol', [{'exit': 0, 'stdout': 'MaxArraySize            = 1001\n'}])
    fake.set_answers('sinfo', [{'exit': 0, 'stdout': 'infinite|32|128000|(null)\n'}])
    checks = doctor.cluster_checks(slurm=SlurmSettings(partition='gpu', max_array_size=1000))
    check = _check(checks=checks, name='slurm-array')
    assert check.status == 'ok'


def test_time_above_the_partition_limit_fails(fake: FakeSlurm) -> None:
    fake.set_answers('sinfo', [{'exit': 0, 'stdout': '2-00:00:00|32|128000|(null)\n'}])
    checks = doctor.cluster_checks(slurm=SlurmSettings(partition='gpu', time='3-00:00:00'))
    check = _check(checks=checks, name='slurm-partition')
    assert check.status == 'fail'
    assert 'time 3-00:00:00 is above the limit of partition gpu (2-00:00:00)' in check.message


def test_fits_on_the_larger_group(fake: FakeSlurm) -> None:
    fake.set_answers(
        'sinfo',
        [{'exit': 0, 'stdout': 'infinite|8|16000|gpu:v100:2\ninfinite|16|32000|gpu:v100:4\n'}],
    )
    checks = doctor.cluster_checks(slurm=SlurmSettings(partition='gpu', gpus=4, cpus_per_task=2))
    check = _check(checks=checks, name='slurm-partition')
    assert check.status == 'ok'


def test_time_and_resources_scattered_across_groups_fails(fake: FakeSlurm) -> None:
    fake.set_answers(
        'sinfo',
        [{'exit': 0, 'stdout': '2-00:00:00|32|128000|gpu:v100:4\n3-00:00:00|32|4000|(null)\n'}],
    )
    checks = doctor.cluster_checks(
        slurm=SlurmSettings(partition='gpu', time='3-00:00:00', gpus=4, cpus_per_task=1)
    )
    check = _check(checks=checks, name='slurm-partition')
    assert check.status == 'fail'


def test_missing_gpus_of_the_type_fails(fake: FakeSlurm) -> None:
    fake.set_answers('sinfo', [{'exit': 0, 'stdout': 'infinite|16|32000|gpu:v100:4\n'}])
    checks = doctor.cluster_checks(
        slurm=SlurmSettings(partition='gpu', gpus=4, cpus_per_task=1, gpu_type='a100')
    )
    check = _check(checks=checks, name='slurm-partition')
    assert check.status == 'fail'
    assert check.message == 'no node of partition gpu has 4 GPUs of type a100; the largest has 0'


def test_memory_is_skipped_without_memory_tracking(fake: FakeSlurm) -> None:
    fake.set_answers('scontrol', [{'exit': 0, 'stdout': 'SelectTypeParameters    = CR_Core\n'}])
    fake.set_answers('sinfo', [{'exit': 0, 'stdout': 'infinite|4|1000|(null)\n'}])
    checks = doctor.cluster_checks(
        slurm=SlurmSettings(partition='gpu', cpus_per_task=4, mem_per_cpu='4G')
    )
    check = _check(checks=checks, name='slurm-partition')
    assert check.status == 'ok'


def test_memory_too_large_fails_with_memory_tracking(fake: FakeSlurm) -> None:
    fake.set_answers(
        'scontrol', [{'exit': 0, 'stdout': 'SelectTypeParameters    = CR_Core_Memory\n'}]
    )
    fake.set_answers('sinfo', [{'exit': 0, 'stdout': 'infinite|4|1000|(null)\n'}])
    settings = SlurmSettings(partition='gpu', cpus_per_task=4, mem_per_cpu='4G')
    checks = doctor.cluster_checks(slurm=settings)
    check = _check(checks=checks, name='slurm-partition')
    assert check.status == 'fail'
    assert (
        f'no node of partition gpu has {settings.requested_memory} bytes of memory' in check.message
    )
    assert f'the largest has {1000 * 1024**2}' in check.message


def test_memory_is_not_checked_when_scontrol_fails(fake: FakeSlurm) -> None:
    fake.set_answers('scontrol', [{'exit': 1, 'stderr': 'boom'}])
    fake.set_answers('sinfo', [{'exit': 0, 'stdout': 'infinite|4|1000|(null)\n'}])
    settings = SlurmSettings(partition='gpu', cpus_per_task=4, mem_per_cpu='4G')
    checks = doctor.cluster_checks(slurm=settings)
    check = _check(checks=checks, name='slurm-partition')
    assert check.status == 'warn'
    assert (
        check.message == f'scontrol did not answer; the memory requirement of '
        f'{settings.requested_memory} bytes is not checked'
    )


def test_time_still_fails_when_scontrol_fails(fake: FakeSlurm) -> None:
    fake.set_answers('scontrol', [{'exit': 1, 'stderr': 'boom'}])
    fake.set_answers('sinfo', [{'exit': 0, 'stdout': '2-00:00:00|32|128000|(null)\n'}])
    checks = doctor.cluster_checks(slurm=SlurmSettings(partition='gpu', time='3-00:00:00'))
    check = _check(checks=checks, name='slurm-partition')
    assert check.status == 'fail'
    assert 'time 3-00:00:00 is above the limit of partition gpu (2-00:00:00)' in check.message


def test_garbage_cpu_count_warns(fake: FakeSlurm) -> None:
    fake.set_answers('sinfo', [{'exit': 0, 'stdout': '2-00:00:00|abc|256000|gpu:a100:4\n'}])
    checks = doctor.cluster_checks(slurm=SlurmSettings(partition='gpu'))
    check = _check(checks=checks, name='slurm-partition')
    assert check.status == 'warn'
    assert 'the node sizes are not checked' in check.message


def test_garbage_time_warns(fake: FakeSlurm) -> None:
    fake.set_answers('sinfo', [{'exit': 0, 'stdout': 'abc|64|256000|(null)\n'}])
    checks = doctor.cluster_checks(slurm=SlurmSettings(partition='gpu'))
    check = _check(checks=checks, name='slurm-partition')
    assert check.status == 'warn'
    assert 'the node sizes are not checked' in check.message


def test_memory_is_not_checked_when_sinfo_reports_none(fake: FakeSlurm) -> None:
    fake.set_answers(
        'scontrol', [{'exit': 0, 'stdout': 'SelectTypeParameters    = CR_Core_Memory\n'}]
    )
    fake.set_answers('sinfo', [{'exit': 0, 'stdout': 'infinite|4|(null)|(null)\n'}])
    settings = SlurmSettings(partition='gpu', cpus_per_task=4, mem_per_cpu='4G')
    checks = doctor.cluster_checks(slurm=settings)
    check = _check(checks=checks, name='slurm-partition')
    assert check.status == 'warn'
    assert (
        check.message == f'sinfo reports no usable memory for partition gpu; '
        f'the memory requirement of {settings.requested_memory} bytes is not checked'
    )


def test_unreported_memory_of_an_ineligible_group_does_not_warn(fake: FakeSlurm) -> None:
    fake.set_answers(
        'scontrol', [{'exit': 0, 'stdout': 'SelectTypeParameters    = CR_Core_Memory\n'}]
    )
    fake.set_answers(
        'sinfo', [{'exit': 0, 'stdout': 'infinite|4|1000|(null)\n2-00:00:00|4|(null)|(null)\n'}]
    )
    settings = SlurmSettings(partition='gpu', time='3-00:00:00', cpus_per_task=4, mem_per_cpu='4G')
    checks = doctor.cluster_checks(slurm=settings)
    check = _check(checks=checks, name='slurm-partition')
    assert check.status == 'fail'
    assert (
        check.message == f'no node of partition gpu holds one job of 4 CPUs, '
        f'{settings.requested_memory} bytes of memory'
    )


def test_known_memory_of_an_ineligible_group_does_not_fail(fake: FakeSlurm) -> None:
    fake.set_answers(
        'scontrol', [{'exit': 0, 'stdout': 'SelectTypeParameters    = CR_Core_Memory\n'}]
    )
    fake.set_answers(
        'sinfo', [{'exit': 0, 'stdout': 'infinite|4|(null)|(null)\ninfinite|2|8192|(null)\n'}]
    )
    settings = SlurmSettings(partition='gpu', cpus_per_task=4, mem_per_cpu='4G')
    checks = doctor.cluster_checks(slurm=settings)
    check = _check(checks=checks, name='slurm-partition')
    assert check.status == 'warn'
    assert (
        check.message == f'sinfo reports no usable memory for partition gpu; '
        f'the memory requirement of {settings.requested_memory} bytes is not checked'
    )


def test_missing_sinfo_warns(
    fake: FakeSlurm, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bin_path = tmp_path / 'no-sinfo'
    bin_path.mkdir()
    for name in ('sbatch', 'scancel', 'sacct', 'squeue', 'scontrol'):
        (bin_path / name).symlink_to(tmp_path / 'slurm' / name)
    monkeypatch.setenv('PATH', str(bin_path))
    checks = doctor.cluster_checks(slurm=SlurmSettings(partition='gpu'))
    check = _check(checks=checks, name='slurm-partition')
    assert check.status == 'warn'


def test_profile_without_slurm_table_warns() -> None:
    checks = doctor.cluster_checks(slurm=None)
    assert len(checks) == 1
    assert checks[0].status == 'warn'
    assert 'no [slurm] table' in checks[0].message

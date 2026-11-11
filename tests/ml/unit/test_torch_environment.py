import subprocess
import sys

import pytest
import torch

from tslab.lab.core.meta.run import collect_hardware, collect_versions
from tslab.lab.core.seeding import seed_everything
from tslab.ml.torch_environment import torch_hardware


def test_importing_ml_adds_torch_seed_and_metadata() -> None:
    import tslab.ml  # noqa: PLC0415
    import tslab.ml.lightning  # noqa: F401, PLC0415 # why: a second import must not register twice

    seed_everything(seed=3)
    first = torch.rand(1)
    seed_everything(seed=3)
    assert torch.equal(first, torch.rand(1))
    assert 'torch' in collect_versions()
    assert 'gpus' in collect_hardware()


def test_importing_ml_twice_registers_the_seeder_once() -> None:
    code = (
        'import tslab.ml, tslab.ml.lightning\n'
        'from tslab.lab.core import seeding\n'
        'assert len(seeding._seeders) == 1\n'
    )
    subprocess.run([sys.executable, '-c', code], check=True)  # noqa: S603 # why: fixed code


def test_torch_hardware_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)

    hardware = torch_hardware()

    assert hardware['cuda_available'] is False
    assert hardware['gpus'] == []
    assert hardware['cuda_visible_devices'] is None


def test_torch_hardware_lists_visible_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda index: f'GPU {index}')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2,4')

    hardware = torch_hardware()

    assert hardware['cuda_available'] is True
    assert hardware['gpus'] == ['GPU 0', 'GPU 1']
    assert hardware['cuda_visible_devices'] == '2,4'

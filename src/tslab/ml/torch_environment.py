"""Torch seeding and torch, Lightning and CUDA metadata, registered with the lab on import."""

import os

from tslab.lab.core.meta.run import package_version

__all__ = ['seed_torch', 'torch_hardware', 'torch_versions']


def seed_torch(seed: int) -> None:
    """Seed PyTorch on all devices."""
    import torch  # noqa: PLC0415 # why: torch is slow to import

    torch.manual_seed(seed)


def torch_versions() -> dict[str, str | None]:
    """Return the installed versions of ``torch`` and ``lightning``."""
    return {
        'torch': package_version(distribution='torch'),
        'lightning': package_version(distribution='lightning'),
    }


def torch_hardware() -> dict[str, object]:
    """Return the CUDA hardware visible to this process.

    Returns:
        Mapping with keys ``cuda_available``, ``cuda_version``, ``gpus`` and
        ``cuda_visible_devices``.
    """
    import torch  # noqa: PLC0415 # why: torch is slow to import and metadata is read by CLI paths

    cuda_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if cuda_available else 0
    return {
        'cuda_available': cuda_available,
        'cuda_version': torch.version.cuda,
        'gpus': [torch.cuda.get_device_name(index) for index in range(gpu_count)],
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
    }

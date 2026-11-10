"""Your own stages.

``@tslab.lab.stage`` turns a function into a cached, resumable step:

- ``name`` is the folder name of the stage under the run folder;
- ``output`` is the artifact type the function returns;
- ``version`` is part of the fingerprint: bump it after changing the code,
  so results cached by the old code are computed again.

The function takes keyword-only arguments, including ``run``. Its other
arguments (settings and input artifacts) form the fingerprint; when it matches
a finished result, the result is loaded instead of computed. Declaring
``name: tslab.lab.InstanceName | None = None`` lets one run call the stage several
times, once per instance name.
"""

import torch
import tslab.lab
import tslab.ts


@tslab.lab.stage(name='add_noise', output=tslab.lab.Tensors, version=1)
def add_noise(
    *,
    run: tslab.lab.Run,
    name: tslab.lab.InstanceName | None = None,
    representations: tslab.lab.Tensors,
    std: float,
) -> tslab.lab.Tensors:
    """Add Gaussian noise to the test representations.

    A small robustness check: a probe fitted on clean training
    representations is then scored on noisy test representations.

    Args:
        run: The current run; its seed seeds the noise.
        name: Instance name, one per noise level.
        representations: From ``tslab.ts.stages.encode``.
        std: Standard deviation of the noise.

    Returns:
        The same tensors with noise added to ``test_representations``.
    """
    del name
    generator = torch.Generator().manual_seed(run.seed)
    tensors = dict(representations.tensors)
    clean = tensors['test_representations']
    noise = torch.randn(clean.shape, generator=generator, dtype=clean.dtype)
    tensors['test_representations'] = clean + std * noise
    return tslab.lab.Tensors(tensors=tensors)

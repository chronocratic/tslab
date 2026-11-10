"""Your own stages.

``@tslab.lab.stage`` turns a function into a cached, resumable step:

- ``name`` is the folder name of the stage under the run folder;
- ``output`` is the artifact type the function returns;
- ``version`` is part of the fingerprint: bump it after changing the code,
  so results cached by the old code are computed again.

The function takes keyword-only arguments, including ``run``. Its other
arguments (settings and input artifacts) form the fingerprint; when it matches
a finished result, the result is loaded instead of computed.
"""

import tslab.lab


@tslab.lab.stage(name='measure', output=tslab.lab.Metrics, version=1)
def measure(*, run: tslab.lab.Run, scale: float) -> tslab.lab.Metrics:
    """Compute one number from the setting and the seed of the run.

    Replace the body with your own work.

    Args:
        run: The current run; its seed is part of the result.
        scale: A setting of the experiment.

    Returns:
        A metric named ``value``.
    """
    return tslab.lab.Metrics(data={'value': scale * (run.seed + 1)})

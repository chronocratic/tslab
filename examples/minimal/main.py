"""A minimal TSLab pipeline: three plain Python stages on synthetic series.

1. ``make_series`` draws noisy sine waves (once per run).
2. ``smooth`` applies a moving average, once per window length (one instance each).
3. ``score`` measures how close each smoothed series is to the clean signal.

Run it, then run it again: the second run loads every result instead of
computing it. Add a window length and only the new instances run::

    uv run python examples/minimal/main.py
    uv run python examples/minimal/main.py --windows 3 9 27
    uv run tslab status runs/minimal
"""

from dataclasses import dataclass, field
import math
from pathlib import Path

import torch

import tslab.lab
import tslab.ml


@dataclass(frozen=True, kw_only=True)
class MinimalConfig:
    """Settings of the minimal pipeline; every field is a command-line flag.

    Attributes:
        series: Number of series.
        length: Steps per series.
        noise: Standard deviation of the added noise.
        windows: Moving-average window lengths, one ``smooth`` instance each.
        root: Folder holding all runs.
        relative_path: The folder of this run under ``root``.
        seed: Seeds Python, NumPy and torch.
        run: Resume and rerun options.
    """

    series: int = 8
    length: int = 256
    noise: float = 0.3
    windows: tuple[int, ...] = (3, 9)
    root: Path = Path('runs')
    relative_path: Path = Path('minimal')
    seed: int = 0
    run: tslab.lab.RunOptions = field(default_factory=tslab.lab.RunOptions)


@tslab.lab.stage(name='make_series', output=tslab.ml.Tensors)
def make_series(*, run: tslab.lab.Run, series: int, length: int, noise: float) -> tslab.ml.Tensors:
    """Draw ``series`` sine waves with random phases, and a noisy copy of each."""
    generator = torch.Generator().manual_seed(run.seed)
    steps = torch.arange(length, dtype=torch.float32)
    phases = 2 * math.pi * torch.rand(series, 1, generator=generator)
    clean = torch.sin(2 * math.pi * steps / 32 + phases)
    noisy = clean + noise * torch.randn(clean.shape, generator=generator)
    return tslab.ml.Tensors(tensors={'clean': clean, 'noisy': noisy})


@tslab.lab.stage(name='smooth', output=tslab.ml.Tensors)
def smooth(
    *,
    run: tslab.lab.Run,
    name: tslab.lab.InstanceName | None = None,
    data: tslab.ml.Tensors,
    window: int,
) -> tslab.ml.Tensors:
    """Smooth the noisy series with a centered moving average of ``window`` steps."""
    del run, name
    kernel = torch.full((1, 1, window), 1.0 / window)
    noisy = data.tensors['noisy'].unsqueeze(1)
    smoothed = torch.nn.functional.conv1d(noisy, kernel, padding=window // 2)
    return tslab.ml.Tensors(tensors={'smoothed': smoothed.squeeze(1)[:, : noisy.shape[-1]]})


@tslab.lab.stage(name='score', output=tslab.lab.Metrics)
def score(
    *,
    run: tslab.lab.Run,
    name: tslab.lab.InstanceName | None = None,
    data: tslab.ml.Tensors,
    smoothed: tslab.ml.Tensors,
) -> tslab.lab.Metrics:
    """Return the mean squared error between the smoothed and the clean series."""
    del run, name
    error = (smoothed.tensors['smoothed'] - data.tensors['clean']).pow(2).mean()
    return tslab.lab.Metrics(data={'mse': float(error)})


def main(*, config: MinimalConfig) -> dict[int, float]:
    """Run the pipeline and return the error per window length."""
    with tslab.lab.Run(config=config, root=config.root, relative_path=config.relative_path) as run:
        data = make_series(run=run, series=config.series, length=config.length, noise=config.noise)
        errors: dict[int, float] = {}
        for window in config.windows:
            smoothed = smooth(run=run, name=f'w{window}', data=data, window=window)
            metrics = score(run=run, name=f'w{window}', data=data, smoothed=smoothed)
            errors[window] = metrics.values['mse']
    return errors


if __name__ == '__main__':
    for window, error in main(config=tslab.lab.parse_config(config_class=MinimalConfig)).items():
        print(f'window {window}: mse {error:.4f}')

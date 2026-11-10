"""A pure-Python three-stage pipeline used by the integration tests.

Stages append one line per execution to ``<root>/calls.txt`` so tests can
check, across processes, exactly which instances executed.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
import time

import torch

import tslab.lab
import tslab.ml


@dataclass(frozen=True, kw_only=True)
class ToyConfig:
    root: Path = Path('runs')
    relative_path: Path = Path('toy')
    size: int = 4
    scales: tuple[float, ...] = (1.0, 2.0)
    fail_scale: float | None = None
    run: tslab.lab.RunOptions = field(default_factory=tslab.lab.RunOptions)
    seed: int = 0


def record_call(*, run: tslab.lab.Run, instance: str) -> None:
    calls = run.directory.parent / 'calls.txt'
    with calls.open(mode='a', encoding='utf-8') as handle:
        handle.write(f'{instance}\n')
    if os.environ.get('TOY_SLEEP_IN') == instance:
        time.sleep(60)


def read_calls(*, root: Path) -> list[str]:
    calls = root / 'calls.txt'
    return calls.read_text(encoding='utf-8').splitlines() if calls.exists() else []


@tslab.lab.stage(name='make_data', output=tslab.ml.Tensors)
def make_data(*, run: tslab.lab.Run, size: int) -> tslab.ml.Tensors:
    record_call(run=run, instance='make_data')
    return tslab.ml.Tensors(tensors={'values': torch.rand(size)})


@tslab.lab.stage(name='transform', output=tslab.ml.Tensors)
def transform(
    *,
    run: tslab.lab.Run,
    name: tslab.lab.InstanceName | None = None,
    data: tslab.ml.Tensors,
    scale: float,
    fail: bool,
) -> tslab.ml.Tensors:
    record_call(run=run, instance=f'transform[{name}]')
    if fail:
        message = f'scale {scale} fails on purpose'
        raise RuntimeError(message)
    return tslab.ml.Tensors(tensors={'values': data.tensors['values'] * scale})


@tslab.lab.stage(name='summarize', output=tslab.lab.Metrics)
def summarize(
    *, run: tslab.lab.Run, name: tslab.lab.InstanceName | None = None, data: tslab.ml.Tensors
) -> tslab.lab.Metrics:
    record_call(run=run, instance=f'summarize[{name}]')
    return tslab.lab.Metrics(data={'mean': float(data.tensors['values'].mean())})


def main(*, config: ToyConfig) -> dict[str, tslab.lab.Artifact]:
    results: dict[str, tslab.lab.Artifact] = {}
    with tslab.lab.Run(config=config, root=config.root, relative_path=config.relative_path) as run:
        data = make_data(run=run, size=config.size)
        for scale in config.scales:
            name = f'x{scale:g}'
            transformed = transform(
                run=run, name=name, data=data, scale=scale, fail=scale == config.fail_scale
            )
            results[name] = summarize(run=run, name=name, data=transformed)
    return results


if __name__ == '__main__':
    main(config=tslab.lab.parse_config(config_class=ToyConfig))

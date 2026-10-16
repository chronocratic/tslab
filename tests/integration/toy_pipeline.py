"""A pure-Python three-stage pipeline used by the integration tests.

Stages append one line per execution to ``<root>/calls.txt`` so tests can
check, across processes, exactly which instances executed.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
import time

import torch

import tslab


@dataclass(frozen=True)
class ToyConfig:
    root: Path = Path('runs')
    relative_path: Path = Path('toy')
    size: int = 4
    scales: tuple[float, ...] = (1.0, 2.0)
    fail_scale: float | None = None
    run: tslab.RunOptions = field(default_factory=tslab.RunOptions)
    seed: int = 0


def record_call(*, run: tslab.Run, instance: str) -> None:
    calls = run.directory.parent / 'calls.txt'
    with calls.open(mode='a', encoding='utf-8') as handle:
        handle.write(f'{instance}\n')
    if os.environ.get('TOY_SLEEP_IN') == instance:
        time.sleep(60)


def read_calls(*, root: Path) -> list[str]:
    calls = root / 'calls.txt'
    return calls.read_text(encoding='utf-8').splitlines() if calls.exists() else []


@tslab.stage(name='make_data', output=tslab.Tensors)
def make_data(*, run: tslab.Run, size: int) -> tslab.Tensors:
    record_call(run=run, instance='make_data')
    return tslab.Tensors(tensors={'values': torch.rand(size)})


@tslab.stage(name='transform', output=tslab.Tensors)
def transform(
    *,
    run: tslab.Run,
    name: tslab.InstanceName | None = None,
    data: tslab.Tensors,
    scale: float,
    fail: bool,
) -> tslab.Tensors:
    record_call(run=run, instance=f'transform[{name}]')
    if fail:
        message = f'scale {scale} fails on purpose'
        raise RuntimeError(message)
    return tslab.Tensors(tensors={'values': data.tensors['values'] * scale})


@tslab.stage(name='summarize', output=tslab.Metrics)
def summarize(
    *, run: tslab.Run, name: tslab.InstanceName | None = None, data: tslab.Tensors
) -> tslab.Metrics:
    record_call(run=run, instance=f'summarize[{name}]')
    return tslab.Metrics(data={'mean': float(data.tensors['values'].mean())})


def main(*, config: ToyConfig) -> dict[str, tslab.Artifact]:
    results: dict[str, tslab.Artifact] = {}
    with tslab.Run(config=config, root=config.root, relative_path=config.relative_path) as run:
        data = make_data(run=run, size=config.size)
        for scale in config.scales:
            name = f'x{scale:g}'
            transformed = transform(
                run=run, name=name, data=data, scale=scale, fail=scale == config.fail_scale
            )
            results[name] = summarize(run=run, name=name, data=transformed)
    return results


if __name__ == '__main__':
    main(config=tslab.parse_config(config_class=ToyConfig))

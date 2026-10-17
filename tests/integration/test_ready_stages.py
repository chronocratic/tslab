from collections.abc import Generator
import contextlib
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

from lightning_toy import FAST
import pytest
import torch
from ts_toy import SineClassification, SineForecasting, ToySequenceModel, ToyVectorModel

import tslab
from tslab.core.status import InstanceState
from tslab.lightning import LightningTrainer
from tslab.probes import RidgeProbe, SVMProbe
from tslab.tasks import forecasting_representations

ENCODER = tslab.ModelAsEncoder(device='cpu')


@dataclass(frozen=True)
class Config:
    seed: int = 0
    run: tslab.RunOptions = field(default_factory=tslab.RunOptions)


@contextlib.contextmanager
def run_in(*, root: Path) -> Generator[tslab.Run]:
    with tslab.Run(config=Config(), root=root, relative_path=Path('r')) as run:
        yield run


pytestmark = pytest.mark.slow


def test_classification_runs_end_to_end_and_reloads(tmp_path: Path) -> None:
    def pipeline() -> tslab.Metrics:
        trainer = LightningTrainer(config=FAST)
        model = ToySequenceModel(in_channels=1, num_classes=2)
        data = SineClassification()
        with run_in(root=tmp_path) as run:
            checkpoint = tslab.stages.train(run=run, trainer=trainer, model=model, data=data)
            representations = tslab.stages.encode(
                run=run,
                trainer=trainer,
                model=model,
                checkpoint=checkpoint,
                data=data,
                encoder=ENCODER,
            )
            return tslab.stages.evaluate(
                run=run, representations=representations, downstream_model=SVMProbe(n_jobs=1)
            )

    first = pipeline()
    assert first.values['accuracy'] >= 0.8
    meta = tmp_path / 'r' / 'encode' / 'default' / 'meta.json'
    before = meta.stat().st_mtime_ns
    second = pipeline()
    assert second.values == first.values
    assert meta.stat().st_mtime_ns == before


def forecast(*, root: Path, model: ToyVectorModel, window_length: int | None) -> dict[int, float]:
    trainer = LightningTrainer(config=FAST)
    data = SineForecasting()
    with run_in(root=root) as run:
        checkpoint = tslab.stages.train(run=run, trainer=trainer, model=model, data=data)
        by_horizon = forecasting_representations(
            run=run,
            trainer=trainer,
            model=model,
            checkpoint=checkpoint,
            data=data,
            encoder=ENCODER,
            horizons=(4, 8),
            window_length=window_length,
            stride=4,
        )
        return {
            horizon: tslab.stages.evaluate(
                run=run,
                name=('forecasting', f'h{horizon}'),
                representations=representations,
                downstream_model=RidgeProbe(n_jobs=1),
            ).values['mse_loss']
            for horizon, representations in by_horizon.items()
        }


def test_forecasting_runs_end_to_end_on_both_paths(tmp_path: Path) -> None:
    dilated = forecast(
        root=tmp_path / 'dilated',
        model=ToySequenceModel(in_channels=2, target_channel=1),
        window_length=None,
    )
    windowed = forecast(
        root=tmp_path / 'windowed',
        model=ToyVectorModel(in_channels=2, target_channel=1),
        window_length=16,
    )
    for errors in (dilated, windowed):
        assert sorted(errors) == [4, 8]
        assert all(0.0 <= error < 1.0 for error in errors.values())


def test_a_time_limit_fails_train_and_the_rerun_resumes(tmp_path: Path) -> None:
    def pipeline(*, time_limit: str | None) -> tuple[tslab.Metrics, dict[str, InstanceState]]:
        # One second, the limit's minimum granularity: a zero limit fires before the
        # first epoch's validation, so Lightning writes no checkpoint to resume from.
        # The large epoch budget plus patience keeps the 1 s cut before early stopping.
        trainer = LightningTrainer(
            config=dataclasses.replace(
                FAST, max_training_time=time_limit, max_epochs=500, early_stopping_patience=300
            )
        )
        model = ToySequenceModel(in_channels=1, num_classes=2)
        data = SineClassification()
        with run_in(root=tmp_path) as run:
            checkpoint = tslab.stages.train(run=run, trainer=trainer, model=model, data=data)
            representations = tslab.stages.encode(
                run=run,
                trainer=trainer,
                model=model,
                checkpoint=checkpoint,
                data=data,
                encoder=ENCODER,
            )
            metrics = tslab.stages.evaluate(
                run=run, representations=representations, downstream_model=SVMProbe(n_jobs=1)
            )
            states = {key: record.state for key, record in run.status().instances.items()}
        return metrics, states

    _, states = pipeline(time_limit='00:00:01')
    assert states['train[default]'] is InstanceState.FAILED
    assert states['encode[default]'] is InstanceState.SKIPPED
    assert (tmp_path / 'r' / 'train' / 'default' / '.incomplete').is_file()
    metrics, states = pipeline(time_limit=None)
    assert set(states.values()) == {InstanceState.DONE}
    last = torch.load(
        f=tmp_path / 'r' / 'train' / 'default' / 'last.ckpt', map_location='cpu', weights_only=False
    )
    # One session cut by the time limit, plus the resumed one.
    assert last['timer_callback_state']['sessions_count'] == 2
    assert 'accuracy' in metrics.values

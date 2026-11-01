from pathlib import Path

import pytest
import torch

from tests.unit.stage_fakes import FakeTrainer, LinearEncoder, run_in, ToySplits
from tests.unit.stages.helpers import encoded
import tslab
from tslab.stages import check_representations


def test_encode_never_uses_the_training_loaders(tmp_path: Path) -> None:
    with run_in(root=tmp_path) as run:
        representations = encoded(run=run)
    shapes = {key: tuple(value.shape) for key, value in representations.tensors.items()}
    assert shapes == {
        'train_representations': (40, 4),
        'train_targets': (40,),
        'val_representations': (20, 4),
        'val_targets': (20,),
        'test_representations': (20, 4),
        'test_targets': (20,),
    }
    assert torch.equal(
        representations.tensors['test_targets'], ToySplits().labeled_split(split='test')[1]
    )


def test_dead_encoder_fails_the_instance(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='non-finite'), run_in(root=tmp_path) as run:
        encoded(run=run, model=LinearEncoder(dead=True))


def test_check_representations_rejects_wrong_shapes() -> None:
    with pytest.raises(ValueError, match='2-D'):
        check_representations(
            representations=torch.zeros(3, 2, 2), expected_rows=3, description='x'
        )
    with pytest.raises(ValueError, match='rows'):
        check_representations(representations=torch.zeros(2, 4), expected_rows=3, description='x')
    check_representations(representations=torch.zeros(0, 4), expected_rows=0, description='x')
    some_nan = torch.zeros(10, 4)
    some_nan[0] = torch.nan
    check_representations(representations=some_nan, expected_rows=10, description='x')


def test_encode_requires_a_fingerprintable_encoder(tmp_path: Path) -> None:
    model, trainer = LinearEncoder(), FakeTrainer()
    # why: the train call sets up the encode call that must fail
    with pytest.raises(TypeError, match='encoder'), run_in(root=tmp_path) as run:  # noqa: PT012
        checkpoint = tslab.stages.train(run=run, trainer=trainer, model=model, data=ToySplits())
        tslab.stages.encode(
            run=run,
            trainer=trainer,
            model=model,
            checkpoint=checkpoint,
            data=ToySplits(),
            encoder=lambda model: model,
        )

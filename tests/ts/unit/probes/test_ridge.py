import pytest

from tests.unit.probes.helpers import ridge_data
from tslab.probes import RidgeProbe


@pytest.mark.parametrize(('val_is_zero', 'alpha'), [(True, 1000), (False, 0.1)])
def test_the_ridge_alpha_is_chosen_on_the_validation_split(
    *, val_is_zero: bool, alpha: float
) -> None:
    probe = RidgeProbe(n_jobs=1)
    probe.fit(**ridge_data(val_is_zero=val_is_zero))
    assert probe.pipeline.named_steps['ridge'].alpha == alpha


def test_the_ridge_caps_train_and_validation() -> None:
    probe = RidgeProbe(max_samples=10, n_jobs=1)
    probe.fit(**ridge_data(val_is_zero=False))
    assert probe.pipeline.named_steps['scaler'].n_samples_seen_ == 20

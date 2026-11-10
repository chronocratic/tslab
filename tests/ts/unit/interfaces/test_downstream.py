from typing import override

import pytest
import torch

from tslab.ts.interfaces import DownstreamModel


class Constant(DownstreamModel):
    @override
    def fit(
        self,
        *,
        train_inputs: torch.Tensor,
        train_targets: torch.Tensor,
        val_inputs: torch.Tensor | None,
        val_targets: torch.Tensor | None,
    ) -> None:
        pass

    @override
    def predict(self, *, inputs: torch.Tensor) -> torch.Tensor:
        return torch.zeros(len(inputs))


def test_default_metrics_explain_what_to_do() -> None:
    with pytest.raises(NotImplementedError, match='metrics='):
        Constant().default_metrics(predictions=torch.zeros(2), targets=torch.zeros(2))

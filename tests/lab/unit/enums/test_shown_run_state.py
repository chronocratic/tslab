import pytest

from tslab.lab.enums import RunState, ShownRunState


def test_every_run_state_has_a_shown_state_with_the_same_text() -> None:
    assert {ShownRunState(state.value) for state in RunState} < set(ShownRunState)


def test_shown_run_state_rejects_an_unknown_text() -> None:
    with pytest.raises(ValueError, match="'paused' is not a valid ShownRunState"):
        ShownRunState('paused')

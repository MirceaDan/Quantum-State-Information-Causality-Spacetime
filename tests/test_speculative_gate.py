import numpy as np
import pytest

from relational_dynamics.speculative_gate import BaselineGate, assert_baseline_only


def test_baseline_gate_is_exactly_zero():
    gate = BaselineGate()
    assert gate.q_norm == 0.0
    assert gate.delta_norm == 0.0
    assert gate.speculative_enabled is False


def test_nonzero_q_or_delta_is_rejected_before_experiment():
    with pytest.raises(ValueError, match="Q must be exactly zero"):
        assert_baseline_only(np.array([[1.0]]), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="Delta_munu must be exactly zero"):
        assert_baseline_only(np.zeros((2, 2)), np.ones((2, 2)))

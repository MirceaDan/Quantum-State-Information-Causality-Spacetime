import numpy as np
import pytest

from relational_dynamics.process import FixedOrderQuantumProcess, ProcessMatrix


def test_unitary_kraus_process_is_completely_positive_and_trace_preserving():
    flip = np.array([[0, 1], [1, 0]], dtype=complex)
    process = FixedOrderQuantumProcess.from_kraus((flip,))
    rho = np.array([[1, 0], [0, 0]], dtype=complex)

    np.testing.assert_allclose(process.apply(rho), np.array([[0, 0], [0, 1]], dtype=complex))
    assert process.is_completely_positive
    assert process.trace_preserving_error < 1e-12
    assert process.causal_order == "fixed"


def test_indefinite_order_interface_is_explicitly_unimplemented():
    with pytest.raises(NotImplementedError):
        ProcessMatrix.from_array(np.eye(4))

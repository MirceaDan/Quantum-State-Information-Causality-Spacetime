import numpy as np

from relational_dynamics.quantum import HamiltonianConstraintSolver


def test_solver_identifies_exact_kernel_and_projects_state():
    constraint = np.diag([0.0, 1.0, 2.0])
    solver = HamiltonianConstraintSolver(constraint, tolerance=1e-10)
    report = solver.solve(preferred_state=np.array([0.0, 1.0, 0.0], dtype=complex))

    assert report.status == "EXACT_KERNEL"
    assert report.kernel_dimension == 1
    assert report.residual < 1e-12
    np.testing.assert_allclose(report.state, np.array([1.0, 0.0, 0.0], dtype=complex))


def test_solver_handles_degenerate_kernel():
    solver = HamiltonianConstraintSolver(np.diag([0.0, 0.0, 2.0]))
    report = solver.solve(preferred_state=np.array([1.0, 1.0, 0.0], dtype=complex) / np.sqrt(2))

    assert report.status == "EXACT_KERNEL"
    assert report.kernel_dimension == 2
    assert report.residual < 1e-12


def test_solver_reports_no_physical_state_without_kernel():
    solver = HamiltonianConstraintSolver(np.diag([1.0, 2.0]))
    report = solver.solve()

    assert report.status == "NO_PHYSICAL_STATE"
    assert report.kernel_dimension == 0
    assert report.state is None
    assert report.residual is None


def test_solver_distinguishes_nonzero_eigenstate_from_constraint_solution():
    solver = HamiltonianConstraintSolver(np.diag([0.0, 1.0]))
    report = solver.solve(preferred_state=np.array([0.0, 1.0], dtype=complex))

    assert report.status == "EXACT_KERNEL"
    assert report.residual < 1e-12
    assert not np.allclose(report.state, np.array([0.0, 1.0]))

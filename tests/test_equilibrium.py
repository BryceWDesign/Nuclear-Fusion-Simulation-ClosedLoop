import numpy as np

from closedloop.equilibrium import manufactured_solution_error, solve_solovev_class_equilibrium


def test_manufactured_solution_is_recovered():
    error, result = manufactured_solution_error(25, 25)
    assert error < 1e-7
    assert result.residual_linf < 1e-6


def test_manufactured_solution_is_grid_independent_to_solver_tolerance():
    e1, r1 = manufactured_solution_error(17, 17)
    e2, r2 = manufactured_solution_error(33, 33)
    # This manufactured solution is polynomial and the discrete operator represents
    # it almost exactly, so truncation error is already below the iterative tolerance.
    assert e1 < 1e-7
    assert e2 < 1e-7
    assert r1.residual_linf < 1e-6
    assert r2.residual_linf < 1e-6


def test_solovev_class_equilibrium_solves_nontrivial_flux():
    result = solve_solovev_class_equilibrium(nR=25, nZ=25)
    assert np.max(np.abs(result.psi_Wb_rad)) > 0
    assert result.residual_linf < 1e-3


def test_solovev_boundaries_are_dirichlet_zero():
    result = solve_solovev_class_equilibrium(nR=17, nZ=17)
    psi = result.psi_Wb_rad
    assert np.allclose(psi[0, :], 0)
    assert np.allclose(psi[-1, :], 0)
    assert np.allclose(psi[:, 0], 0)
    assert np.allclose(psi[:, -1], 0)
